import os
import sys
import subprocess
import argparse
import json
from multiprocessing import Process
from typing import Any, List, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def repo_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)

def model_path(model_name: str) -> str:
    local_path = repo_path("models", model_name)
    if os.path.exists(local_path):
        return local_path
    afs_path = os.path.join("/mnt/afs/models", model_name)
    if os.path.exists(afs_path):
        return afs_path
    return local_path

def default_models(agent_type: str) -> List[str]:
    if agent_type == "no_memory":
        return [
            model_path("Qwen3-1.7B"),
            model_path("Qwen3-4B-Instruct"),
            repo_path("output", "NoMemory-GRPO-Qwen3-1.7B", "checkpoint_250"),
            repo_path("output", "NoMemory-GRPO-Qwen3-4B-Instruct", "checkpoint_250"),
        ]
    return [
        model_path("Qwen3-1.7B"),
        model_path("Qwen3-4B-Instruct"),
        repo_path("output", "MemAgent-RL-Qwen3-1.7B", "checkpoint_250"),
        repo_path("output", "MemAgent-RL-Qwen3-4B-Instruct", "checkpoint_250"),
    ]

def is_task_completed(task: Dict[str, Any]) -> bool:
    """Check if the task has already been successfully completed."""
    out_file = task['out_file']
    if not os.path.exists(out_file):
        return False
    
    try:
        with open(out_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        summary = data.get('summary')
        if not summary:
            return False
            
        # Check model and dataset match
        if summary.get('model') != task['model']:
            return False
        if summary.get('dataset') != task['dataset']:
            return False

        for key in [
            'agent_type',
            'n_paths',
            'chunk_size',
            'max_prompt_length',
            'max_tokens',
            'context_truncation_side',
        ]:
            if summary.get(key) != task.get(key):
                return False
            
        # Check current_accuracy is valid number
        acc = summary.get('current_accuracy')
        if not isinstance(acc, (int, float)):
            return False
            
        # Check processed == total
        processed = summary.get('processed')
        total = summary.get('total')
        if processed is None or total is None or processed != total:
            return False
            
        return True
    except Exception:
        return False

def run_tasks_on_gpu(gpu_id: str, tasks: List[Dict[str, Any]]):
    """
    Worker process function to run a queue of tasks sequentially on a specific GPU.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    for t in tasks:
        print(f"\n[GPU {gpu_id}] Starting task:")
        print(f"  Model:   {t['model']}")
        print(f"  Dataset: {t['dataset']}")
        
        worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate_worker.py")
        cmd = [
            sys.executable, worker_script,
            "--model_path", t['model'],
            "--dataset_path", t['dataset'],
            "--output_file", t['out_file'],
            "--agent_type", t['agent_type'],
            "--chunk_size", str(t['chunk_size']),
            "--n_paths", str(t['n_paths']),
            "--max_prompt_length", str(t['max_prompt_length']),
            "--max_tokens", str(t['max_tokens']),
            "--context_truncation_side", t['context_truncation_side'],
            "--vllm_max_model_len", str(t['vllm_max_model_len']),
            "--gpu_memory_utilization", str(t['gpu_memory_utilization']),
        ]
        
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"[GPU {gpu_id}] Finished task: {t['out_file']}")
        except subprocess.CalledProcessError as e:
            print(f"[GPU {gpu_id}] Task failed with error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Orchestrator for Evaluation")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs to use. Defaults to CUDA_VISIBLE_DEVICES if set, otherwise 0,1,2,3,4,5,6,7.")
    parser.add_argument("--agent_type", type=str, default="all", choices=["memagent", "no_memory", "all"], help="Evaluation policy. Use 'all' to run both memagent and no_memory in one pass.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model/checkpoint paths. Ignored when agent_type=all.")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset paths.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for evaluation results")
    parser.add_argument("--chunk_size", type=int, default=2500)
    parser.add_argument("--n_paths", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=12000)
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument("--context_truncation_side", type=str, default="head", choices=["head", "tail", "middle"])
    parser.add_argument("--vllm_max_model_len", type=int, default=32768)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    gpu_arg = args.gpus
    if gpu_arg is None:
        gpu_arg = os.environ.get("CUDA_VISIBLE_DEVICES") or "0,1,2,3,4,5,6,7"
    gpus = [g.strip() for g in gpu_arg.split(",") if g.strip()]
    num_gpus = len(gpus)

    if args.agent_type == "all":
        type_model_pairs = [
            (at, m)
            for at in ["memagent", "no_memory"]
            for m in default_models(at)
        ]
    elif args.models:
        type_model_pairs = [(args.agent_type, m.strip()) for m in args.models.split(",") if m.strip()]
    else:
        type_model_pairs = [(args.agent_type, m) for m in default_models(args.agent_type)]

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = [
            repo_path("datas", "eval_50.json"),
            repo_path("datas", "eval_100.json"),
            repo_path("datas", "eval_200.json"),
            repo_path("datas", "eval_400.json"),
            repo_path("datas", "eval_fwe_16384.json"),
            repo_path("datas", "eval_fwe_32768.json"),
        ]

    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")
    os.makedirs(output_dir, exist_ok=True)

    # Generate all tasks
    all_tasks = []
    tasks = []
    for (agent_type, m) in type_model_pairs:
        if not os.path.exists(m):
            print(f"Skipping missing model/checkpoint: {m}")
            continue
        for d in datasets:
            if not os.path.exists(d):
                print(f"Skipping missing dataset: {d}")
                continue
            parts = m.rstrip("/").split("/")
            m_name = f"{parts[-2]}_{parts[-1]}"
            d_name = os.path.basename(d).replace(".json", "")
            out_file = os.path.join(output_dir, f"{agent_type}_{m_name}_{d_name}.json")
            
            task_info = {
                "model": m,
                "dataset": d,
                "out_file": out_file,
                "agent_type": agent_type,
                "chunk_size": args.chunk_size,
                "n_paths": args.n_paths,
                "max_prompt_length": args.max_prompt_length,
                "max_tokens": args.max_tokens,
                "context_truncation_side": args.context_truncation_side,
                "vllm_max_model_len": args.vllm_max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            }
            all_tasks.append(task_info)
            
            if is_task_completed(task_info):
                print(f"Skipping completed task: {out_file}")
            else:
                tasks.append(task_info)

    # Distribute tasks across GPUs
    gpu_queues = {g: [] for g in gpus}
    for i, t in enumerate(tasks):
        gpu = gpus[i % num_gpus]
        gpu_queues[gpu].append(t)

    print(f"Total tasks: {len(tasks)}")
    print(f"Total GPUs: {num_gpus}")
    for gpu, q in gpu_queues.items():
        print(f"  GPU {gpu}: {len(q)} tasks")

    # Spawn processes
    processes = []
    for gpu, q in gpu_queues.items():
        if not q:
            continue
        p = Process(target=run_tasks_on_gpu, args=(gpu, q))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("\nAll evaluations finished! Results are saved in the 'eval_results' directory.")
    
    # Optional: summarize results
    print("\n--- Summary ---")
    for t in all_tasks:
        if os.path.exists(t['out_file']):
            try:
                with open(t['out_file'], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    acc = data.get('summary', {}).get('current_accuracy', 0.0)
                    print(f"{t['out_file']}: Accuracy = {acc:.4f}")
            except Exception as e:
                print(f"Failed to read {t['out_file']}: {e}")

if __name__ == "__main__":
    main()
