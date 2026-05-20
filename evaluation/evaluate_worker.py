import os
import json
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import sys
# Make sure we can import memfactory utils
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from memfactory.common.utils import TEMPLATE, TEMPLATE_FINAL_BOXED, evaluate_memory_agent, LLMClient
from memfactory.common.utils import build_no_memory_qa_prompt

def process_dataset(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    if len(data) > 0:
        first = data[0]
        # Check dataset structure based on difference investigation
        if 'outputs' in first and 'answer_prefix' in first:
            # eval_fwe_xx type dataset
            for item in data:
                assert isinstance(item['outputs'], list) and len(item['outputs']) == 3, "Each item must have a list of 3 outputs"
                # 把 outputs 中的 3 个元素拼起来, 用逗号空格分隔
                ground_truth = ', '.join(item['outputs'])
                samples.append({
                    'question': item['input'],
                    'context': item['context'],
                    'ground_truth': ground_truth
                })
        else:
            # eval_xx type dataset
            for item in data:
                ground_truth = item['answers'][0] if isinstance(item['answers'], list) else item['answers']
                samples.append({
                    'question': item['input'],
                    'context': item['context'],
                    'ground_truth': ground_truth
                })
    return samples

def main():
    parser = argparse.ArgumentParser(description="Evaluate a specific model on a specific dataset using vLLM.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model or base model name")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset json file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the evaluation results")
    parser.add_argument("--agent_type", type=str, default="memagent", choices=["memagent", "no_memory"], help="Evaluation policy")
    parser.add_argument("--chunk_size", type=int, default=2500, help="Token length for context chunking")
    parser.add_argument("--n_paths", type=int, default=4, help="Number of reasoning paths (N=4)")
    parser.add_argument("--max_prompt_length", type=int, default=8192, help="Prompt length budget for direct no-memory evaluation")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum generated tokens per model call")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--context_truncation_side", type=str, default="head", choices=["head", "tail", "middle"], help="How to truncate direct no-memory context")
    parser.add_argument("--vllm_max_model_len", type=int, default=32768, help="vLLM max_model_len to avoid over-reserving KV cache")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="vLLM GPU memory utilization")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_path}")
    samples = process_dataset(args.dataset_path)
    
    print(f"Loading model: {args.model_path}")
    # Initialize vLLM (tensor_parallel_size=1, since we allocate 1 GPU per task)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=args.vllm_max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    stop_token_ids = [tokenizer.eos_token_id]
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.pad_token_id)

    # Initialize sampling params: n=1 because we provide a list of distinct prompts at each step
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop_token_ids=stop_token_ids
    )
    
    # Initialize LLM judge
    llm_client = LLMClient()
    
    results = []
    total_score = 0.0
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {os.path.basename(args.dataset_path)}")):
        question = sample['question']
        context = sample['context']
        ground_truth = sample['ground_truth']
        
        context_ids = tokenizer.encode(context, add_special_tokens=False)

        if args.agent_type == "no_memory":
            formatted_final_prompts = [
                build_no_memory_qa_prompt(
                    tokenizer=tokenizer,
                    question=question,
                    context_ids=context_ids,
                    max_prompt_length=args.max_prompt_length,
                    context_truncation_side=args.context_truncation_side,
                    apply_chat_template=True,
                )
                for _ in range(args.n_paths)
            ]
            final_outputs = llm.generate(formatted_final_prompts, sampling_params, use_tqdm=False)
            final_responses = [out.outputs[0].text for out in final_outputs]
        else:
            total_length = len(context_ids)
            num_chunks = (total_length + args.chunk_size - 1) // args.chunk_size

            # Initialize memories for N paths
            memories = ["No previous memory"] * args.n_paths

            # Iterative Context Processing
            for step in range(num_chunks):
                start_idx = step * args.chunk_size
                end_idx = min((step + 1) * args.chunk_size, total_length)
                chunk_text = tokenizer.decode(context_ids[start_idx:end_idx], skip_special_tokens=True)

                prompts = [TEMPLATE.format(prompt=question, memory=memories[j], chunk=chunk_text) for j in range(args.n_paths)]
                msgs_list = [[{"role": "user", "content": p}] for p in prompts]
                formatted_prompts = [tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs_list]

                outputs = llm.generate(formatted_prompts, sampling_params, use_tqdm=False)

                for j in range(args.n_paths):
                    response_text = outputs[j].outputs[0].text
                    memories[j] = response_text
                    # Extract memory thought process
                    if "<think>" in response_text:
                        if "</think>" in response_text:
                            memories[j] = response_text.split("</think>")[-1].strip()
                        else:
                            memories[j] = response_text[-100:].strip()

            # Final Answer Generation
            final_prompts = [TEMPLATE_FINAL_BOXED.format(prompt=question, memory=memories[j]) for j in range(args.n_paths)]
            msgs_list = [[{"role": "user", "content": p}] for p in final_prompts]
            formatted_final_prompts = [tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs_list]

            final_outputs = llm.generate(formatted_final_prompts, sampling_params, use_tqdm=False)
            final_responses = [out.outputs[0].text for out in final_outputs]
        
        # Evaluation using LLM as a judge
        path_scores = []
        for j in range(args.n_paths):
            score = evaluate_memory_agent(final_responses[j], ground_truth, question, llm_client=llm_client)
            path_scores.append(score)
            
        avg_score = sum(path_scores) / args.n_paths
        total_score += avg_score
        
        results.append({
            'question': question,
            'ground_truth': ground_truth,
            'final_responses': final_responses,
            'path_scores': path_scores,
            'avg_score': avg_score
        })
        
        # Save intermediate results
        summary = {
            'model': args.model_path,
            'dataset': args.dataset_path,
            'agent_type': args.agent_type,
            'n_paths': args.n_paths,
            'chunk_size': args.chunk_size,
            'max_prompt_length': args.max_prompt_length,
            'max_tokens': args.max_tokens,
            'context_truncation_side': args.context_truncation_side,
            'current_accuracy': total_score / (i + 1),
            'processed': i + 1,
            'total': len(samples)
        }
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'results': results}, f, ensure_ascii=False, indent=2)

    final_accuracy = total_score / len(samples)
    print(f"\nFinal Accuracy for {args.model_path} on {args.dataset_path}: {final_accuracy:.4f}\n")

if __name__ == "__main__":
    main()
