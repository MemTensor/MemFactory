import os
import sys
import json
import argparse
import torch
import gc
import re
import copy
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
rl_dir = os.path.join(project_root, "RL")
sys.path.append(project_root)
sys.path.append(rl_dir)

try:
    import mem_utils
    from src.common import MemoryItem, ConversationMessage, get_memory_store, generate_id, MemoryStore
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class ModelInference:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        print(f"Loading model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 2048) -> str:
        return self.generate_batch([prompt], max_new_tokens)[0]

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 2048) -> List[str]:
        if not prompts:
            return []
            
        messages_list = [[{"role": "user", "content": p}] for p in prompts]
        texts = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages_list]
        
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, padding_side="left").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0, # Deterministic for eval
                do_sample=False
            )
            
        input_len = inputs.input_ids.shape[1]
        generated_texts = []
        for output in outputs:
            generated_ids = output[input_len:]
            generated_texts.append(self.tokenizer.decode(generated_ids, skip_special_tokens=True))
            
        return generated_texts

class LocomoEvaluator:
    def __init__(self, model: ModelInference, data_path: str, evaluator: mem_utils.MemoryEvaluator):
        self.model = model
        self.evaluator = evaluator
        self.store = evaluator.store
        self.data_path = data_path
        
        # Load Data
        self.sample = self._load_sample()
        
        # Preprocess QAs
        self.valid_qas = self._preprocess_qas(self.sample)
        
        # Runtime State
        self.memory_session_map = {} # memory_id -> session_idx
        self.dialogue_buffer = [] # List[ConversationMessage]
        self.dia_id_buffer = [] # List[str]
        self.BUFFER_SIZE = 4
        
        # Metrics
        self.total_qas = 0
        self.correct_qas = 0
        self.qa_details = []

    def _load_sample(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Return the first sample (locomo10.json usually contains one sample or a list)
            if isinstance(data, list):
                return data[-1]
            return data

    def _parse_dia_id(self, dia_id: str) -> Tuple[int, int]:
        match = re.match(r"D(\d+):(\d+)", dia_id)
        if match:
            return int(match.group(1)), int(match.group(2))
        return 0, 0

    def _preprocess_qas(self, sample: Dict) -> List[Dict]:
        qas = sample.get('qa', [])
        valid_qas = []
        
        for qa in qas:
            # Filter adversarial
            if qa.get('adversarial_answer'):
                continue
            
            ans_str = qa.get('answer')
            if not ans_str:
                continue
            
            qa['final_answer'] = ans_str
            
            evidence = qa.get('evidence', [])
            if not evidence:
                continue
            
            parsed_ev = [self._parse_dia_id(e) for e in evidence]
            parsed_ev = [e for e in parsed_ev if e[0] > 0]
            
            if not parsed_ev:
                continue
                
            sessions = [e[0] for e in parsed_ev]
            min_s, max_s = min(sessions), max(sessions)
            
            # Rule: If span > 3 sessions, ignore.
            if (max_s - min_s) > 2:
                continue
                
            parsed_ev.sort()
            trigger = parsed_ev[-1] # (session, turn)
            
            qa['trigger'] = trigger
            qa['processed'] = False
            qa['min_evidence_session'] = min_s
            valid_qas.append(qa)
            
        valid_qas.sort(key=lambda x: x['trigger'])
        print(f"Loaded {len(valid_qas)} valid QAs")
        return valid_qas

    def _get_session_content(self, session_idx: int) -> List[Dict]:
        key = f"session_{session_idx}"
        if 'conversation' in self.sample and key in self.sample['conversation']:
            return self.sample['conversation'][key]
        return []

    def _get_session_time(self, session_idx: int) -> str:
        """Extracts and formats session time (e.g. '2023-05-08')."""
        key = f"session_{session_idx}_date_time"
        if 'conversation' in self.sample and key in self.sample['conversation']:
            time_str = self.sample['conversation'][key]
            # Format: "1:56 pm on 8 May, 2023"
            try:
                dt = datetime.strptime(time_str, "%I:%M %p on %d %B, %Y")
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                return time_str
        assert False, "必须要有正确的时间戳"
        return ""

    def _prune_memories(self, current_session_idx: int):
        """Keep only memories from [current_session_idx - 2, current_session_idx]"""
        to_delete = []
        for mid, s_idx in self.memory_session_map.items():
            if s_idx < current_session_idx - 2:
                to_delete.append(mid)
        
        if to_delete:
            # print(f"Pruning {len(to_delete)} memories older than session {current_session_idx - 2}")
            for mid in to_delete:
                self.store.delete(mid)
                del self.memory_session_map[mid]

    def _custom_apply_update(self, context_memory: List[Dict], update_plan: Dict, extraction_output: str, current_session_idx: int):
        """
        Modified apply_update_plan to track session IDs
        """
        if not update_plan or "operations" not in update_plan:
            return

        _, _, id_map = mem_utils.prepare_memory_lists(context_memory, extraction_output)
        
        for op in update_plan["operations"]:
            temp_id = op.get("id")
            action = op.get("op", "NONE").upper()
            
            if temp_id not in id_map:
                continue
                
            origin_type, origin_obj = id_map[temp_id]
            
            if origin_type == "context":
                if action == "DEL":
                    self.store.delete(origin_obj.id)
                    if origin_obj.id in self.memory_session_map:
                        del self.memory_session_map[origin_obj.id]
            
            elif origin_type == "candidate":
                if action in ["ADD", "UPDATE"]:
                    key = op.get("key", origin_obj.get("key"))
                    value = op.get("value", origin_obj.get("value"))
                    
                    new_mem = MemoryItem(
                        id=generate_id(),
                        key=key,
                        value=value,
                        memory_type=origin_obj.get("memory_type", "UserMemory"),
                        tags=origin_obj.get("tags", [])
                    )
                    self.store.save(new_mem)
                    # Track session
                    self.memory_session_map[new_mem.id] = current_session_idx

    def _evaluate_qa(self, qa: Dict):
        query = qa['question']
        answer = qa['final_answer']
        
        # Retrieval
        retrieved_docs = self.evaluator.retrieve(query, top_k=15)
        context_str = "\n".join([f"- {m.key}: {m.value}" for m in retrieved_docs])
        
        # QA Generation
        qa_prompt = mem_utils.QA_PROMPT.format(context=context_str, question=query)
        pred_answer = self.evaluator.llm.chat("You are a helpful assistant.", qa_prompt)
        
        # Clean think tags
        if "<think>" in pred_answer:
            if "</think>" in pred_answer:
                pred_answer = pred_answer.split("</think>")[-1].strip()
            else:
                pred_answer = pred_answer[-100:].strip()

        # Judge
        judge_prompt = mem_utils.JUDGE_PROMPT.format(question=query, answer=answer, prediction=pred_answer)
        judge_result = self.evaluator.llm.chat("You are an impartial judge.", judge_prompt)
        
        is_correct = False
        if "True" in judge_result:
            is_correct = True
        
        self.total_qas += 1
        if is_correct:
            self.correct_qas += 1
            
        self.qa_details.append({
            "question": query,
            "answer": answer,
            "prediction": pred_answer,
            "is_correct": is_correct,
            "judge_raw": judge_result
        })
        # print(f"  QA: {query[:30]}... -> {'Correct' if is_correct else 'Wrong'}")

    def run(self):
        # Determine max session
        max_session = 0
        if 'conversation' in self.sample:
            for k in self.sample['conversation']:
                if k.startswith('session_'):
                    try:
                        idx = int(k.split('_')[1])
                        max_session = max(max_session, idx)
                    except:
                        pass
        
        print(f"Processing {max_session} sessions...")
        
        # Reset Store (Just in case)
        if self.store.use_mock:
            self.store.neo4j._mock_store.clear()
            self.store.neo4j._mock_edges = []
            self.store.milvus._mock_vectors.clear()
        
        # Iterate sessions
        for s_idx in range(1, max_session + 1):
            # 1. Prune old memories
            self._prune_memories(s_idx)
            
            turns = self._get_session_content(s_idx)
            session_time = self._get_session_time(s_idx)
            if not turns:
                continue
                
            # print(f"Session {s_idx}: {len(turns)} turns")
            
            for turn in tqdm(turns, desc=f"Session {s_idx}", leave=False):
                # Update buffers
                msg = ConversationMessage(
                    role=turn.get('speaker', 'user'),
                    content=turn.get('text', ''),
                    timestamp=turn.get('timestamp', session_time)
                )
                dia_id = turn.get('dia_id', '')
                
                self.dialogue_buffer.append(msg)
                self.dia_id_buffer.append(dia_id)
                
                if len(self.dialogue_buffer) > self.BUFFER_SIZE:
                    self.dialogue_buffer.pop(0)
                    self.dia_id_buffer.pop(0)
                
                # --- MEMORY PROCESSING ---
                
                # 1. Extraction
                # Convert buffer to dict format expected by construct_extraction_prompt
                # construct_extraction_prompt expects list of dicts with role, content
                conv_for_prompt = [{"role": m.role, "content": m.content, "timestamp": m.timestamp} for m in self.dialogue_buffer]
                ext_prompt = mem_utils.construct_extraction_prompt(conv_for_prompt)
                ext_output = self.model.generate(ext_prompt)
                
                # 2. Retrieval & Update Context
                candidates = []
                ext_json = mem_utils.parse_json_from_text(ext_output)
                if ext_json and "memory_list" in ext_json:
                    for m in ext_json["memory_list"]:
                        if isinstance(m, dict):
                            candidates.append(m)
                
                # Retrieve relevant context
                context_memory_items = []
                seen_ids = set()
                
                for cand in candidates:
                    query_text = f"{cand.get('key', '')} {cand.get('value', '')}"
                    results = self.store.search_similar(query_text, top_k=3)
                    for item, _ in results:
                        if item.id not in seen_ids:
                            context_memory_items.append(item.to_dict())
                            seen_ids.add(item.id)
                    if len(context_memory_items) >= 12:
                        break
                        
                # 3. Update Planning
                upd_prompt = mem_utils.construct_update_prompt(context_memory_items, ext_output)
                upd_output = self.model.generate(upd_prompt)
                
                # 4. Apply Update
                upd_json = mem_utils.parse_json_from_text(upd_output)
                self._custom_apply_update(context_memory_items, upd_json, ext_output, s_idx)
                
                # --- QA CHECK ---
                current_parsed_id = self._parse_dia_id(dia_id)
                
                for qa in self.valid_qas:
                    if qa.get('processed'):
                        continue
                        
                    if qa['trigger'] == current_parsed_id:
                        # Check evidence window
                        # We only keep [s_idx-2, s_idx].
                        # Evidence must be >= s_idx-2
                        if qa['min_evidence_session'] >= s_idx - 2:
                            self._evaluate_qa(qa)
                            qa['processed'] = True

        return {
            "accuracy": self.correct_qas / self.total_qas if self.total_qas > 0 else 0.0,
            "correct": self.correct_qas,
            "count": self.total_qas,
            "details": self.qa_details
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--data_path", type=str, default=os.path.join(os.path.dirname(current_dir), "datas/locomo10.json"))
    args = parser.parse_args()
    
    # Models Configuration
    models_config = {
        "Qwen3-1.7B": "/home/models/Qwen3-1.7B",
        "Qwen3-4B": "/home/models/qwen3-4b",
        "4B-100step": "../output/mem_grpo/checkpoint_100",
        "4B-200step": "../output/mem_grpo/checkpoint_200",
        "4B-300step": "../output/mem_grpo/checkpoint_300",
        "4B-400step": "../output/mem_grpo/checkpoint_400",
    }
    
    # Override if CLI provided
    if args.model_name and args.model_path:
        models_config = {args.model_name: args.model_path}
    
    # Initialize Environment
    print("Initializing MemoryEvaluator...")
    try:
        evaluator = mem_utils.MemoryEvaluator()
    except Exception as e:
        print(f"Failed to initialize MemoryEvaluator: {e}")
        return

    results_summary = {}

    for model_name, model_path in models_config.items():
        print(f"\n{'='*20}\nEvaluating Model: {model_name}\n{'='*20}")
        
        # Load Model
        try:
            model_inference = ModelInference(model_path)
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            continue

        # Run Evaluation
        print(f"Starting LoCoMo Evaluation on {args.data_path}")
        locomo_eval = LocomoEvaluator(model_inference, args.data_path, evaluator)
        result = locomo_eval.run()
        
        results_summary[model_name] = result
        print(f"Result for {model_name}: Accuracy {result['accuracy']:.2%}")
        
        # Save detailed results per model
        output_file = f"locomo_results_{model_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        # Cleanup
        del model_inference
        del locomo_eval
        gc.collect()
        torch.cuda.empty_cache()

    # Final Report
    print("\n" + "="*40)
    print("FINAL EVALUATION REPORT")
    print("="*40)
    for model_name, res in results_summary.items():
        print(f"Model: {model_name}")
        print(f"  Accuracy: {res['accuracy']:.2%} ({res['correct']}/{res['count']})")
    print("="*40)

if __name__ == "__main__":
    main()
