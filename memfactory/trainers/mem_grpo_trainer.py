from dataclasses import dataclass, field
import torch
import torch.distributed as dist
import torch.nn.functional as F
import os
from copy import deepcopy
from contextlib import nullcontext
from typing import Optional, Dict, Any
from tqdm import tqdm
import swanlab
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from ..common.registry import TRAINER_REGISTRY, ENV_REGISTRY, AGENT_REGISTRY
from ..common.utils import LLMClient
from ..modules.base import Samples

@dataclass
class MemGRPOArguments:
    output_dir: str = "./output"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr: float = 5e-7
    save_steps: int = 500
    max_steps: Optional[int] = None
    epoch: int = 1
    beta: float = 0.1
    clip_eps: float = 0.2
    gradient_accumulation_steps: int = 1
    num_iterations: int = 1
    batch_size: int = 1
    train_micro_batch_size: int = 1
    logprob_batch_size: int = 1
    gradient_checkpointing: bool = True
    
    # MemFactory specific
    agent_type: str = "naive"
    env_type: str = "longcontext"
    max_chunk_number: int = 5
    num_generations: int = 4
    generation_batch_size: int = 1
    max_prompt_length: int = 4096
    max_generate_length: int = 2048
    context_truncation_side: str = "head"
    chunk_size: int = 2048
    report_to_swanlab: bool = False
    distributed: bool = False
    local_rank: int = 0
    global_rank: int = 0
    world_size: int = 1
    
    # Training control
    do_shuffle: bool = False
    train_extraction: bool = False
    train_update: bool = False

@TRAINER_REGISTRY.register("mem_grpo")
class MemGRPOTrainer:
    def __init__(self, model, args: MemGRPOArguments, tokenizer, ref_model=None):
        self.args = args
        self.is_distributed = (
            self.args.distributed
            and dist.is_available()
            and dist.is_initialized()
            and self.args.world_size > 1
        )
        self.local_rank = self.args.local_rank
        self.global_rank = self.args.global_rank
        self.world_size = self.args.world_size if self.is_distributed else 1
        self.is_main_process = self.global_rank == 0

        if getattr(model, "hf_device_map", None):
            self.policy_model = model
        else:
            self.policy_model = model.to(self.args.device)
        self.tokenizer = tokenizer
        self._swanlab_failed = False
        
        if self.args.gradient_checkpointing:
            if hasattr(self.policy_model, "config"):
                self.policy_model.config.use_cache = False
            try:
                self.policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.policy_model.gradient_checkpointing_enable()
            
        self.ref_model = ref_model
        if self.ref_model is None and self.args.beta != 0.0:
            self.ref_model = deepcopy(self.policy_model)
            self.ref_model.eval()
            self.ref_model.requires_grad_(False)

        if self.is_distributed:
            self.model = DistributedDataParallel(
                self.policy_model,
                device_ids=[self.local_rank] if self._is_cuda_device() else None,
                output_device=self.local_rank if self._is_cuda_device() else None,
                find_unused_parameters=False,
            )
        else:
            self.model = self.policy_model

        trainable_params = [p for p in self.policy_model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.args.lr)
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler = torch.amp.GradScaler() if (self._is_cuda_device() and self.policy_model.dtype != torch.bfloat16) else None
        
        self.update_steps = 0
        self.llm_client = LLMClient() # For reward computation

    def _is_cuda_device(self) -> bool:
        return isinstance(self.args.device, str) and self.args.device.startswith("cuda")

    def _log(self, log_dict: Dict[str, Any]):
        if not self.is_main_process or not self.args.report_to_swanlab or self._swanlab_failed:
            return
        try:
            swanlab.log(log_dict)
        except Exception as exc:
            self._swanlab_failed = True
            print(f"[SwanLab] Logging disabled after failure: {exc}")

    def _has_trainable_samples(self, samples_output) -> bool:
        if not samples_output:
            return False
        if isinstance(samples_output, Samples):
            samples_dict = {"default": samples_output}
        elif isinstance(samples_output, dict):
            samples_dict = samples_output
        else:
            return False

        for step_type, samples in samples_dict.items():
            should_train = True
            if step_type == 'extraction' and not self.args.train_extraction:
                should_train = False
            if step_type == 'update' and not self.args.train_update:
                should_train = False
            if should_train and samples.rewards is not None:
                return True
        return False

    def _all_ranks_ready(self, local_ready: bool) -> bool:
        if not self.is_distributed:
            return local_ready
        ready = torch.tensor(1 if local_ready else 0, device=self.args.device, dtype=torch.int)
        dist.all_reduce(ready, op=dist.ReduceOp.MIN)
        return bool(ready.item())

    def _batch_global_steps(self, batch: Dict[str, Any]) -> int:
        local_batch = len(batch.get("context_ids", []))
        return max(1, local_batch) * self.world_size

    def _save_due_checkpoints(self, previous_steps: int):
        if not self.is_main_process or self.args.save_steps <= 0:
            return
        next_save = ((previous_steps // self.args.save_steps) + 1) * self.args.save_steps
        while next_save <= self.update_steps:
            self.save_model(f"checkpoint_{next_save}")
            next_save += self.args.save_steps

    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        output = model(input_ids, attention_mask=attention_mask, use_cache=False)
        logits = output.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        per_token_nll = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        log_probs_labels = -per_token_nll.view(shift_labels.size())
        
        return log_probs_labels[:, -num_actions:]

    def compute_loss(self, model, inputs):
        prompt_response_ids = inputs['prompt_response_ids']
        attention_mask = inputs['attention_mask']
        action_mask = inputs['action_mask']
        num_actions = inputs['old_action_log_probs'].size(1)
        
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

    def train_step(self, inputs, step):
        self.model.train()
        training_batch_size = max(1, self.args.train_micro_batch_size)
        total_samples = inputs['prompt_response_ids'].size(0)
        total_loss = 0.0
        should_step_optimizer = (step + 1) % self.args.gradient_accumulation_steps == 0

        for i in range(0, total_samples, training_batch_size):
            end_i = min(i + training_batch_size, total_samples)
            mini_inputs = {k: v[i:end_i] if v is not None else None for k, v in inputs.items()}

            if self.scaler:
                with torch.amp.autocast(device_type='cuda'):
                    loss = self.compute_loss(self.model, mini_inputs)
            else:
                loss = self.compute_loss(self.model, mini_inputs)

            mini_batch_size = end_i - i
            scale_factor = mini_batch_size / total_samples
            scaled_loss = loss * scale_factor

            if self.scaler:
                self.scaler.scale(scaled_loss / self.args.gradient_accumulation_steps).backward()
            else:
                (scaled_loss / self.args.gradient_accumulation_steps).backward()

            total_loss += scaled_loss.item()

        if should_step_optimizer:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            self._log({"train/loss": total_loss, "step": self.update_steps})
            if self.is_main_process and self.update_steps % 10 == 0:
                print(f"Step {self.update_steps}: Loss {total_loss:.4f}")

    def save_model(self, name):
        if not self.is_main_process:
            return
        path = os.path.join(self.args.output_dir, name)
        os.makedirs(path, exist_ok=True)
        self.policy_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    def _prepare_train_inputs(self, samples):
        inference_batch_size = max(1, self.args.logprob_batch_size)
        total_samples = samples.prompt_response_ids.size(0)
        
        all_old_log_probs = []
        all_ref_log_probs = []
        
        with torch.no_grad():
            for i in range(0, total_samples, inference_batch_size):
                end_i = min(i + inference_batch_size, total_samples)
                
                # Slice the batch
                mini_ids = samples.prompt_response_ids[i:end_i]
                mini_mask = samples.attention_mask[i:end_i]
                
                # Compute old log probs
                mini_old_lp = self.get_action_log_probs(self.policy_model, mini_ids, mini_mask, samples.num_actions)
                all_old_log_probs.append(mini_old_lp)
                
                # Compute ref log probs
                if self.ref_model:
                    mini_ref_lp = self.get_action_log_probs(self.ref_model, mini_ids, mini_mask, samples.num_actions)
                    all_ref_log_probs.append(mini_ref_lp)
        
        old_lp = torch.cat(all_old_log_probs, dim=0)
        ref_lp = torch.cat(all_ref_log_probs, dim=0) if self.ref_model else None
        
        return {
            "prompt_response_ids": samples.prompt_response_ids,
            "attention_mask": samples.attention_mask,
            "action_mask": samples.action_mask,
            "advantages": samples.rewards,
            "old_action_log_probs": old_lp,
            "ref_action_log_probs": ref_lp
        }

    def train(self, data_path):
        step_count = 0
        # no_memory agent 需要打乱数据以避免连续相似样本导致方差长期为0；
        # 带 memory 的 agent 训练时保持原来不打乱的行为（与历史一致）
        effective_shuffle = self.args.do_shuffle or (self.args.agent_type == "no_memory")

        # 1. Initialize Env
        EnvClass = ENV_REGISTRY.get(self.args.env_type)
        env = EnvClass(data_path, self.tokenizer)
        # 2. Initialize Agent
        AgentClass = AGENT_REGISTRY.get(self.args.agent_type)
        agent = AgentClass(
            self.tokenizer, 
            device=self.args.device,
            chunk_size=self.args.chunk_size, 
            max_chunk_number=self.args.max_chunk_number,
            num_generations=self.args.num_generations,
            generation_batch_size=self.args.generation_batch_size,
            max_prompt_length=self.args.max_prompt_length,
            max_generate_length=self.args.max_generate_length,
            context_truncation_side=self.args.context_truncation_side,
            report_to_swanlab=self.args.report_to_swanlab
        )

        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                env,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=effective_shuffle,
                drop_last=False,
            )
        dataloader = DataLoader(
            env,
            batch_size=self.args.batch_size,
            shuffle=effective_shuffle if sampler is None else False,
            sampler=sampler,
            collate_fn=env.collate_fn,
        )
        
        # Define reward function wrapper
        # For MemoryBankEnv, it uses: predictions(Dict), ground_truths, num_generations
        # For LongContextMemoryEnv, it uses: predictions(List), ground_truths, questions
        # We need to unify or adapt
        def reward_fn_wrapper(*args, **kwargs):
            return env.compute_reward(*args, **kwargs, llm_client=self.llm_client)

        for epoch in range(self.args.epoch):
            if sampler is not None:
                sampler.set_epoch(epoch)
            pbar = tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                desc=f"Epoch {epoch + 1}/{self.args.epoch}",
                disable=not self.is_main_process,
            )
            for idx, batch in pbar:
                # Rollout
                self.policy_model.eval()
                samples_output = agent.rollout(self.policy_model, batch, reward_fn=reward_fn_wrapper)
                # no_memory agent 的 rollout 不会返回 None（方差为0时 advantages 置零），
                # 无需跨 rank 协调 skip，直接继续。
                # 其他带 memory 的 agent 仍保留原逻辑：只支持单卡，不做 DDP skip 协调。
                if self.args.agent_type != "no_memory":
                    local_ready = self._has_trainable_samples(samples_output)
                    if not local_ready:
                        if self._is_cuda_device():
                            torch.cuda.empty_cache()
                        continue
                
                if samples_output:
                    # Normalize output to Dict[str, Samples]
                    if isinstance(samples_output, Samples):
                        samples_dict = {"default": samples_output}
                    elif isinstance(samples_output, dict):
                        samples_dict = samples_output
                    else:
                        continue
                    
                    log_dict = {}
                    for step_type, samples in samples_dict.items():
                        if samples.rewards is not None:
                            mean_adv = samples.rewards.mean().item()
                            log_dict[f"train/advantage_{step_type}"] = mean_adv
                        if samples.response_length is not None:
                            mean_len = samples.response_length.float().mean().item()
                            log_dict[f"train/response_length_{step_type}"] = mean_len

                    if log_dict and self.is_main_process:
                        reward_parts = ", ".join(
                            f"{k.replace('train/', '')}={v:.4f}"
                            for k, v in log_dict.items()
                        )
                        print(f"[Step {self.update_steps}] {reward_parts}")
                        log_dict["step"] = self.update_steps
                        self._log(log_dict)

                    # Inner Loop
                    for _ in range(self.args.num_iterations):
                        # Iterate over all types of samples returned by rollout
                        for step_type, samples in samples_dict.items():
                            
                            should_train = True
                            if step_type == 'extraction' and not self.args.train_extraction:
                                should_train = False
                            if step_type == 'update' and not self.args.train_update:
                                should_train = False
                            if should_train and samples.rewards is not None:
                                train_inputs = self._prepare_train_inputs(samples)
                                self.train_step(train_inputs, step_count)
                                step_count += 1
                    
                    previous_steps = self.update_steps
                    self.update_steps += self._batch_global_steps(batch)
                    self._save_due_checkpoints(previous_steps)
                    if self.args.max_steps is not None and self.args.max_steps > 0 and self.update_steps >= self.args.max_steps:
                        if self.is_main_process:
                            print(f"Reached max_steps={self.args.max_steps}; stopping training.")
                        return
                
                if self._is_cuda_device():
                    torch.cuda.empty_cache()
