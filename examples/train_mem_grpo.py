import argparse
import os
import sys
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
import swanlab

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memfactory.trainers.mem_grpo_trainer import MemGRPOTrainer, MemGRPOArguments
import memfactory.envs  # Register envs
import memfactory.modules # Register modules
import memfactory.agents # Register agents

def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun DDP requires CUDA for this training script.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    return distributed, local_rank, global_rank, world_size

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def main():
    distributed, local_rank, global_rank, world_size = setup_distributed()
    is_main_process = global_rank == 0

    parser = HfArgumentParser((MemGRPOArguments,))
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    cli_parser.add_argument("--data_path", type=str, required=True, help="Path to the training data")
    cli_parser.add_argument("--wandb_name", type=str, default=None, help="SwanLab experiment name")
    cli_parser.add_argument("--swanlab_project", type=str, default="MemFactory", help="SwanLab project name")
    cli_parser.add_argument("--disable_swanlab", action="store_true", help="Disable SwanLab even if wandb_name is set")
    
    # We parse known args for CLI, and the rest for MemGRPOArguments
    args, remaining_args = cli_parser.parse_known_args()
    
    # Parse MemGRPOArguments from remaining_args
    grpo_args = parser.parse_args_into_dataclasses(args=remaining_args)[0]
    grpo_args.distributed = distributed
    grpo_args.local_rank = local_rank
    grpo_args.global_rank = global_rank
    grpo_args.world_size = world_size
    if distributed:
        grpo_args.device = f"cuda:{local_rank}"
    
    if is_main_process:
        print(f"Loading model from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True
    }

    try:
        import accelerate
        model_kwargs["low_cpu_mem_usage"] = True
    except ImportError:
        pass
    
    use_flash_attention = False
    if torch.cuda.is_available():
        try:
            import flash_attn
            use_flash_attention = True
        except ImportError:
            pass

    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        if is_main_process:
            print("Using Flash Attention 2")
    else:
        if is_main_process:
            print("Flash Attention 2 not found, using default attention")
        
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **model_kwargs
        )
    except Exception as exc:
        if model_kwargs.get("attn_implementation") == "flash_attention_2":
            if is_main_process:
                print(f"Flash Attention 2 load failed ({exc}); retrying with default attention")
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                **model_kwargs
            )
        else:
            raise
    
    # Init SwanLab
    grpo_args.report_to_swanlab = False
    if is_main_process and args.wandb_name and not args.disable_swanlab:
        swanlab_api_key = os.getenv("SWANLAB_API_KEY")
        if not swanlab_api_key:
            print("SWANLAB_API_KEY is not set; SwanLab logging is disabled for this run.")
        else:
            try:
                swanlab.init(
                    project=args.swanlab_project,
                    config={**vars(grpo_args), "model_name_or_path": args.model_name_or_path, "data_path": args.data_path},
                    name=args.wandb_name
                )
                grpo_args.report_to_swanlab = True
            except Exception as exc:
                print(f"SwanLab init failed ({exc}); continuing without SwanLab logging.")
    
    try:
        if is_main_process:
            print("Initializing Trainer...")
        trainer = MemGRPOTrainer(
            model=model,
            args=grpo_args,
            tokenizer=tokenizer
        )
        
        if is_main_process:
            print(f"Starting Training with Agent: {grpo_args.agent_type}, Env: {grpo_args.env_type}...")
            if distributed:
                print(f"DDP enabled: world_size={world_size}, max_steps counts global rollout samples.")
        trainer.train(args.data_path)
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
