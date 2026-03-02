import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional, Union, Any, Dict, Tuple
from copy import deepcopy
import json
import os
import swanlab
import random
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd

# Import mem_utils and src.common
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import mem_utils
# Ensure we can import src.common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import MemoryItem, generate_id

@dataclass
class Samples:
    prompt_response_ids: torch.Tensor
    response_ids: torch.Tensor
    prompt: Any # Can be string or list of strings
    answer: Any
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    response_length: torch.Tensor
    prompt_length: torch.Tensor
    step_type: str # 'extraction' or 'update'
    rewards: Optional[torch.Tensor] = None

@dataclass
class MemGRPOArguments:
    output_dir: str = "./output"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr: float = 5e-7
    save_steps: int = 500
    epoch: int = 1
    num_generations: int = 4 # Group size
    max_prompt_length: int = 4096
    max_generate_length: int = 2048
    beta: float = 0.1 # KL penalty
    clip_eps: float = 0.2
    gradient_accumulation_steps: int = 1
    num_iterations: int = 1
    batch_size: int = 1
    train_extraction: bool = True
    train_update: bool = True
    gradient_checkpointing: bool = True

class MemoryDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = []
        if os.path.exists(data_path):
            try:
                df = pd.read_parquet(data_path)
                for _, row in df.iterrows():
                    context = row['context']
                    extra_info = row['extra_info']
                    prompt = row['prompt']
                    reward_model = row['reward_model']
                    
                    # Pre-encode context as requested
                    context_ids = tokenizer.encode(context, add_special_tokens=False)
                    
                    self.data.append({
                        'context_ids': context_ids,
                        'context': context,
                        'question': extra_info.get('question', ''),
                        'extra_info': extra_info,
                        'prompt': prompt,
                        'ground_truth': reward_model.get('ground_truth', ''),
                        'reward_model': reward_model
                    })
            except Exception as e:
                print(f"Error loading parquet file: {e}")
        else:
            print(f"Warning: {data_path} not found.")
            assert False, f"{data_path} not found."
        
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class MemGRPOTrainer:
    def __init__(self,
                 model,
                 args: MemGRPOArguments,
                 train_dataset: Dataset,
                 tokenizer,
                 ref_model=None):
        self.args = args
        self.model = model.to(self.args.device)
        
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        self.tokenizer = self.get_tokenizer(tokenizer)
        
        self.ref_model = ref_model
        if self.ref_model is None and self.args.beta != 0.0:
            self.ref_model = deepcopy(model)
            self.ref_model.eval()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_dataset = train_dataset
        self.input_buffer = [None] * self.args.gradient_accumulation_steps
        self.update_steps = 0
        self.global_steps = 0
        # BFloat16 does not need GradScaler
        self.scaler = torch.amp.GradScaler() if (self.args.device == 'cuda' and self.model.dtype != torch.bfloat16) else None
        
        # Initialize Evaluator
        self.evaluator = mem_utils.MemoryEvaluator()

    def get_tokenizer(self, tokenizer):
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _generate_with_pytorch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Fallback generation using PyTorch model.generate"""
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.args.max_generate_length,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
        return outputs

    def run_llm_loop(self, batch_data):
        """
        Implementation of the multi-step memory generation loop.
        Returns a list of (prompt_text, response_text, reward_score) tuples.
        """
        results = []
        
        # Iterate over each sample in the batch (even if batch_size > 1)
        # Note: batch_data is a dict of lists (collate_fn)
        bs = len(batch_data['context_ids'])
        chunk_size = 512 # Define a reasonable chunk size
        max_chunk_number = 8

        for i in range(bs):
            context_ids = batch_data['context_ids'][i] # List[int]
            question = batch_data['question'][i]
            ground_truth = batch_data['ground_truth'][i]
            
            # 1. Setup chunks
            total_length = len(context_ids)
            num_chunks = (total_length + chunk_size - 1) // chunk_size
            assert num_chunks <= max_chunk_number

            # 2. Initialize memories for N generations
            num_generations = self.args.num_generations
            memories = ["No previous memory"] * num_generations
            
            # List of lists to store trajectories
            # trajectories[j] = [(prompt_text, response_text), ...]
            trajectories = [[] for _ in range(num_generations)]
            
            # 3. Iterate through chunks
            for step in range(num_chunks):
                start_idx = step * chunk_size
                end_idx = min((step + 1) * chunk_size, total_length)
                chunk_token_ids = context_ids[start_idx:end_idx]
                chunk_text = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
                
                # Construct prompts for this step for all N generations
                step_prompts = []
                for j in range(num_generations):
                    prompt_text = mem_utils.TEMPLATE.format(
                        prompt=question,
                        memory=memories[j],
                        chunk=chunk_text
                    )
                    step_prompts.append(prompt_text)
                
                # Generate responses (new memories)
                msgs_list = [[{"role": "user", "content": p}] for p in step_prompts]
                formatted_prompts = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs_list]
                
                tokenized = self.tokenizer(formatted_prompts, padding=True, return_tensors='pt').to(self.args.device)
                
                outputs = self._generate_with_pytorch(tokenized['input_ids'], tokenized['attention_mask'])
                
                # Extract responses
                input_len = tokenized['input_ids'].size(1)
                generated_ids = outputs[:, input_len:]
                generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Update memories and store trajectory
                for j in range(num_generations):
                    response_text = generated_texts[j]
                    memories[j] = response_text
                    # Store (prompt, response)
                    trajectories[j].append((formatted_prompts[j], response_text))
            
            # 4. Final Turn
            final_prompts = []
            for j in range(num_generations):
                prompt_text = mem_utils.TEMPLATE_FINAL_BOXED.format(
                    prompt=question,
                    memory=memories[j]
                )
                final_prompts.append(prompt_text)
            
            msgs_list = [[{"role": "user", "content": p}] for p in final_prompts]
            formatted_final_prompts = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs_list]
            
            tokenized = self.tokenizer(formatted_final_prompts, padding=True, return_tensors='pt').to(self.args.device)
            outputs = self._generate_with_pytorch(tokenized['input_ids'], tokenized['attention_mask'])
            
            input_len = tokenized['input_ids'].size(1)
            generated_ids = outputs[:, input_len:]
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # 5. Evaluation and Group Advantage Calculation
            scores = []
            
            for j in range(num_generations):
                final_response = generated_texts[j]
                # Add final turn to trajectory
                trajectories[j].append((formatted_final_prompts[j], final_response))
                
                # Evaluate
                score = mem_utils.evaluate_memory_agent(final_response, ground_truth)
                scores.append(score)

            # Convert to tensor for statistics
            scores_tensor = torch.tensor(scores, dtype=torch.float32, device=self.args.device)
            mean_score = scores_tensor.mean()
            std_score = scores_tensor.std()
            
            # (1) Filter: If std is 0 (or very small), discard this sample
            if std_score.item() < 1e-6:
                continue
                
            # (2) Calculate Advantages (GRPO)
            # Adv_i = (Score_i - Mean) / (Std + epsilon)
            advantages = (scores_tensor - mean_score) / (std_score + 1e-8)
            
            # (3) Broadcast and Collect
            for j in range(num_generations):
                traj_advantage = advantages[j].item()
                traj_steps = trajectories[j]
                
                # Broadcast this advantage to ALL steps in this trajectory
                for p, r in traj_steps:
                    results.append((p, r, traj_advantage))
            
            swanlab.log({
                "reward_mean": mean_score.item(),
                "reward_std": std_score.item()
            })
                    
        return results

    def generate_samples(self, batch_data):
        self.model.eval()
        assert len(batch_data) == 1, "we advice use bs=1 yet"
        # 1. Run LLM Loop to get raw data
        # results: List of (prompt_text, response_text, advantage_score)
        results = self.run_llm_loop(batch_data)
        
        if not results:
            return None

        # 2. Tokenize and Collect Data
        all_input_ids = []
        all_attention_masks = []
        all_action_masks = []
        all_advantages = []
        
        # We need to tokenize again to get lengths for masking
        # Optimization: run_llm_loop could return ids, but text is safer for now
        
        for prompt, response, advantage in results:
            # Tokenize Prompt and Response
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = self.tokenizer.encode(response, add_special_tokens=False)
            
            # Create concatenated sequence
            # Format: [Prompt, Response]
            seq_ids = prompt_ids + response_ids
            
            # Store raw data for padding later
            all_input_ids.append(torch.tensor(seq_ids, dtype=torch.long))
            all_advantages.append(advantage)
            
            # Create Masks (pre-padding)
            # Attention Mask: All 1s for now
            att_mask = torch.ones(len(seq_ids), dtype=torch.long)
            all_attention_masks.append(att_mask)
            
            # Action Mask: 0 for Prompt, 1 for Response
            act_mask = torch.zeros(len(seq_ids), dtype=torch.bool) # BoolTensor
            act_mask[len(prompt_ids):] = True
            all_action_masks.append(act_mask)

        # 3. Global Padding (Left Padding)
        # Pad sequence is: [PAD, ..., PAD, Prompt, Response]
        # This keeps the Prompt+Response contiguity intact at the end
        
        # Use pad_sequence for right padding first, then we might need to flip or handle left padding manually?
        # Actually, torch.nn.utils.rnn.pad_sequence does right padding. 
        # For Left Padding, we can reverse, pad right, then reverse back.
        
        def left_pad_sequence(sequences, batch_first=True, padding_value=0):
            assert batch_first, "if not [bs, len], will cause fatal error!!!"
            reversed_sequences = [seq.flip(0) for seq in sequences]
            padded_reversed = torch.nn.utils.rnn.pad_sequence(reversed_sequences, batch_first=batch_first, padding_value=padding_value)
            return padded_reversed.flip(1)

        pad_token_id = self.tokenizer.pad_token_id
        
        # Pad Input IDs
        padded_input_ids = left_pad_sequence(all_input_ids, batch_first=True, padding_value=pad_token_id).to(self.args.device)
        
        # Pad Attention Mask (0 for Pad)
        padded_attention_mask = left_pad_sequence(all_attention_masks, batch_first=True, padding_value=0).to(self.args.device)
        
        # Pad Action Mask (0/False for Pad)
        # Note: Padding should be False (no loss on pads)
        padded_action_mask = left_pad_sequence(all_action_masks, batch_first=True, padding_value=False).to(self.args.device)
        
        # Advantages to Tensor
        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32, device=self.args.device)
        
        # 4. Construct Single Samples Object
        # Note: We are flattening everything into a single large batch
        samples = Samples(
            prompt_response_ids=padded_input_ids,
            response_ids=None, # Not used in new logic
            prompt=None, # Not used
            answer=None, # Not used
            attention_mask=padded_attention_mask,
            action_mask=padded_action_mask,
            num_actions=padded_action_mask.sum(dim=1), # Number of action tokens
            response_length=padded_action_mask.sum(dim=1),
            prompt_length=padded_attention_mask.sum(dim=1) - padded_action_mask.sum(dim=1),
            step_type='extraction', # Generic type
            rewards=advantages_tensor
        )
            
        return samples

    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        output = model(input_ids, attention_mask=attention_mask, use_cache=False)
        logits = output.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
        return action_log_probs

    def generate_experiences(self, batch_data):
        self.model.eval()
        # samples is now a SINGLE Samples object containing the entire flattened batch
        samples = self.generate_samples(batch_data)
        
        if samples is None:
            return None

        batch_exp = {
            "prompt_response_ids": [],
            "attention_mask": [],
            "action_mask": [],
            "advantages": [],
            "old_action_log_probs": [],
            "ref_action_log_probs": []
        }
        
        # Calculate Log Probs in Mini-batches to save memory
        inference_batch_size = 12 # Adjust based on GPU memory
        total_samples = samples.prompt_response_ids.size(0)
        
        all_old_log_probs = []
        all_ref_log_probs = []
        
        with torch.no_grad():
            for i in range(0, total_samples, inference_batch_size):
                end_i = min(i + inference_batch_size, total_samples)
                
                # Slice the batch
                mini_ids = samples.prompt_response_ids[i:end_i]
                mini_mask = samples.attention_mask[i:end_i]
                mini_num_actions = samples.num_actions[i:end_i]
                
                # Compute old log probs
                mini_old_lp = self.get_action_log_probs(self.model, mini_ids, mini_mask, mini_num_actions)
                all_old_log_probs.append(mini_old_lp)
                
                # Compute ref log probs
                if self.ref_model:
                    mini_ref_lp = self.get_action_log_probs(self.ref_model, mini_ids, mini_mask, mini_num_actions)
                    all_ref_log_probs.append(mini_ref_lp)

        # Concatenate results
        old_log_probs = torch.cat(all_old_log_probs, dim=0)
        ref_log_probs = torch.cat(all_ref_log_probs, dim=0) if self.ref_model else None

        batch_exp["prompt_response_ids"].append(samples.prompt_response_ids)
        batch_exp["attention_mask"].append(samples.attention_mask)
        batch_exp["action_mask"].append(samples.action_mask)
        batch_exp["advantages"].append(samples.rewards) # rewards are advantages
        batch_exp["old_action_log_probs"].append(old_log_probs)
        if ref_log_probs is not None:
            batch_exp["ref_action_log_probs"].append(ref_log_probs)

        # Helper to collate a single experience dict
        def collate_exp(exp_dict):
            if not exp_dict["prompt_response_ids"]:
                return None
            return {
                "prompt_response_ids": torch.cat(exp_dict["prompt_response_ids"], dim=0),
                "attention_mask": torch.cat(exp_dict["attention_mask"], dim=0),
                "action_mask": torch.cat(exp_dict["action_mask"], dim=0),
                "advantages": torch.cat(exp_dict["advantages"], dim=0),
                "old_action_log_probs": torch.cat(exp_dict["old_action_log_probs"], dim=0),
                "ref_action_log_probs": torch.cat(exp_dict["ref_action_log_probs"], dim=0) if self.ref_model else None
            }
        
        # We only use "extraction" key for compatibility with train loop
        return {
            "extraction": collate_exp(batch_exp),
            "update": None
        }

    def compute_loss(self, model, inputs):
        prompt_response_ids = inputs['prompt_response_ids']
        attention_mask = inputs['attention_mask']
        action_mask = inputs['action_mask']
        num_actions = action_mask.size(1)
        
        action_log_probs = self.get_action_log_probs(model, prompt_response_ids, attention_mask, num_actions)
        
        k3 = None
        if self.args.beta != 0.0 and inputs.get('ref_action_log_probs') is not None:
            ref_action_log_probs = inputs['ref_action_log_probs']
            log_ratio = ref_action_log_probs - action_log_probs
            log_ratio = log_ratio * action_mask
            k3 = log_ratio.exp() - 1 - log_ratio
            
        advantages = inputs['advantages']
        old_action_log_probs = inputs['old_action_log_probs']
        
        coef_1 = torch.exp(action_log_probs - old_action_log_probs)
        coef_2 = torch.clamp(coef_1, 1 - self.args.clip_eps, 1 + self.args.clip_eps)
        
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss * action_mask
        
        if k3 is not None:
            per_token_loss = per_token_loss + self.args.beta * k3
            
        loss = per_token_loss.sum(dim=1) / (action_mask.sum(dim=1) + 1e-8)
        return loss.mean()

    def train_step(self, model, inputs, optimizer, step):
        if inputs is None: return
        model.train()
        
        # Mini-batching for training
        training_batch_size = 8 # Adjust based on GPU memory
        total_samples = inputs['prompt_response_ids'].size(0)
        
        total_loss = 0.0
        
        for i in range(0, total_samples, training_batch_size):
            end_i = min(i + training_batch_size, total_samples)
            
            # Slice inputs for mini-batch
            mini_inputs = {
                'prompt_response_ids': inputs['prompt_response_ids'][i:end_i],
                'attention_mask': inputs['attention_mask'][i:end_i],
                'action_mask': inputs['action_mask'][i:end_i],
                'advantages': inputs['advantages'][i:end_i],
                'old_action_log_probs': inputs['old_action_log_probs'][i:end_i],
                'ref_action_log_probs': inputs['ref_action_log_probs'][i:end_i] if inputs['ref_action_log_probs'] is not None else None
            }
            
            # Compute loss for mini-batch
            # Note: compute_loss returns mean loss over the mini-batch
            loss = self.compute_loss(model, mini_inputs)
            
            # Scale loss for gradient accumulation
            # The gradient should be averaged over the WHOLE batch (total_samples), not just the mini-batch.
            # PyTorch's backward() accumulates gradients.
            # If we do loss.backward() for each mini-batch, the gradients will sum up.
            # So we need to scale each mini-batch loss by (mini_batch_size / total_samples) if loss is a mean.
            # Let's verify: 
            # Total Loss = Sum(L_i) / N
            # Mini-batch Loss = Sum(L_j) / M
            # We want backward to produce d(Total Loss)/dw
            # d(Total Loss)/dw = (1/N) * Sum(dL_i/dw)
            # Our loop produces Sum(d(Mini-Loss)/dw) = Sum( (1/M) * Sum(dL_j/dw) )
            # If M is constant, this is (1/M) * Sum(dL_i/dw) = (N/M) * d(Total Loss)/dw
            # So we need to multiply by (M/N).
            
            mini_batch_size = end_i - i
            scale_factor = mini_batch_size / total_samples
            
            scaled_loss = loss * scale_factor
            
            if self.scaler:
                 with torch.amp.autocast(device_type='cuda'):
                     # Further divide by gradient_accumulation_steps for global accumulation
                     self.scaler.scale(scaled_loss / self.args.gradient_accumulation_steps).backward()
            else:
                 (scaled_loss / self.args.gradient_accumulation_steps).backward()
            
            total_loss += scaled_loss.item()
             
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            if self.scaler:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

            swanlab.log({
                            "train/loss": total_loss,
                            "step": self.update_steps
                        })
            
            if self.update_steps % 10 == 0:
                print(f"Step {self.update_steps}: Loss {total_loss:.4f}")

    def train(self):
        self.optimizer.zero_grad()
        self.global_steps = self.args.num_iterations * self.args.epoch * len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps)
        
        # Custom collate function to handle variable length lists (don't stack them)
        def collate_fn(batch):
            keys = batch[0].keys()
            return {key: [d[key] for d in batch] for key in keys}

        for epoch in range(self.args.epoch):
            
            dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=False, collate_fn=collate_fn)
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{self.args.epoch}")
            for idx, batch in pbar:
                experiences = self.generate_experiences(batch)
                
                pbar.set_postfix(step=idx, global_step=self.update_steps)
                
                if experiences:
                    # Inner Loop for GRPO/PPO
                    for _ in range(self.args.num_iterations):
                        # Train Extraction (Used as the main training loop)
                        if experiences["extraction"] is not None:
                            self.train_step(self.model, experiences["extraction"], self.optimizer, idx)
                        
                        # Note: 'update' key is now None, so we skip it
                    
                    self.update_steps += 1
                    
                    if self.update_steps % self.args.save_steps == 0:
                        self.save_model(f"checkpoint_{self.update_steps}")
                
                torch.cuda.empty_cache()

    def save_model(self, name):
        path = os.path.join(self.args.output_dir, name)
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/home/models/qwen3-4b", help="Path to the model")
    parser.add_argument("--data_path", type=str, default="/home/guozl/project/MemRL/Memory-CookBook/MemAgent/data/hotpotqa_dev.parquet", help="Path to the training data")
    parser.add_argument("--output_dir", type=str, default="./output/mem_grpo", help="Output directory")
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty beta")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save steps")
    parser.add_argument("--max_generate_length", type=int, default=4096, help="Max generate length")
    parser.add_argument("--wandb_name", type=str, default=None, help="SwanLab experiment name")
    
    # Add training mode arguments
    parser.add_argument("--no_train_extraction", action="store_true", help="Disable training extraction")
    parser.add_argument("--no_train_update", action="store_true", help="Disable training update")
    
    # Parse known args to allow passing other args if needed
    args, unknown = parser.parse_known_args()
    
    print(f"Loading model from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        "device_map": "auto",
        "trust_remote_code": True
    }
    
    try:
        import flash_attn
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")
    except ImportError:
        print("Flash Attention 2 not found, using default attention")
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        **model_kwargs
    )
    
    # Configure Training Arguments
    grpo_args = MemGRPOArguments(
        output_dir=args.output_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        beta=args.beta,
        num_generations=4, # Group size
        save_steps=args.save_steps,
        epoch=2,
        max_prompt_length=3072,
        max_generate_length=args.max_generate_length,
        train_extraction=not args.no_train_extraction,
        train_update=not args.no_train_update
    )
    os.environ["SWANLAB_API_KEY"] = "Zkrggz0kWlnEuNRu5r4dz"
    swanlab.init(
            project="MemFactory",
            config=vars(grpo_args),
            name=args.wandb_name
        )
    
    print(f"Loading data from {args.data_path}...")
    # Initialize Dataset
    dataset = MemoryDataset(args.data_path, tokenizer)
    
    if len(dataset) == 0:
        print("Error: Dataset is empty or file not found. Please run 'python scripts/process_locomo.py' first.")
        assert False
        
    print("Initializing Trainer...")
    trainer = MemGRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    
    print("Starting Training...")
    trainer.train()
