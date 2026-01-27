
import json
import os
import sys
import math
from typing import List, Dict, Any

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.common import (
    MemoryItem, 
    get_llm_client, 
    get_memory_store,
    format_conversation,
    ConversationMessage
)
from RL.mem_utils import (
    construct_extraction_prompt,
    parse_json_from_text
)

def process_data(source_path: str, target_path: str, k: int = 8, max_items: int = 512):
    print(f"Processing data from {source_path} to {target_path}")
    
    # Initialize components
    llm = get_llm_client()
    store = get_memory_store()
    
    # Load source data
    with open(source_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    # Load existing target data to skip
    processed_ids = set()
    if os.path.exists(target_path):
        with open(target_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        if "sample_id" in item:
                            processed_ids.add(item["sample_id"])
                    except json.JSONDecodeError:
                        continue
    
    print(f"Found {len(processed_ids)} already processed items.")
    
    count = 0
    with open(target_path, 'a', encoding='utf-8') as f_out:
        for item in source_data:
            if count >= max_items:
                break
                
            sample_id = item.get("sample_id")
            if sample_id in processed_ids:
                continue
            
            print(f"Processing item {sample_id}...")
            
            # 1. Initialize Memory Store with existing memories (M)
            # We use mock mode for per-sample isolation
            store.use_mock = True
            store.from_list(item["M"]) # Load M into store
            # 2. Extract from last 2 turns of f
            facts = item["f"]
            if not facts:
                print(f"Skipping {sample_id}: No facts.")
                continue
                
            # Take last 2 turns
            recent_facts = facts[-2:]
            
            # Construct extraction prompt
            # Note: construct_extraction_prompt expects dicts with role/content/timestamp usually, 
            # but checks helper function implementation.
            # mem_utils.py: construct_extraction_prompt iterates and creates ConversationMessage
            # It expects 'role' and 'content' keys in the input list.
            extraction_prompt = construct_extraction_prompt(recent_facts)
            
            # Call LLM
            extraction_output = llm.chat("You are a memory extraction expert.", extraction_prompt)
            ext_json = parse_json_from_text(extraction_output)
            
            extracted_memories = []
            if ext_json and "memory_list" in ext_json:
                extracted_memories = ext_json["memory_list"]
            
            # 3. Retrieve similar memories
            retrieved_context = []
            
            if not extracted_memories:
                # Fallback: maybe use the query or recent facts? 
                # User says "if extract...". If nothing extracted, context_memory is empty.
                pass
            else:
                # Strategy:
                # If 1 extracted memory -> retrieve k
                # If >= 2 extracted memories -> take last 2, retrieve k/2 for each
                
                search_targets = []
                if len(extracted_memories) == 1:
                    search_targets.append((extracted_memories[0], k))
                else:
                    # Take last 2
                    targets = extracted_memories[-2:]
                    k_per = k // 2 # Integer division
                    for t in targets:
                        search_targets.append((t, k_per))
                
                seen_ids = set()
                
                for mem_dict, limit in search_targets:
                    # Construct query from key + value
                    query_text = f"{mem_dict.get('key', '')}: {mem_dict.get('value', '')}"
                    
                    # Search
                    results = store.search_similar(query_text, top_k=limit)
                    
                    for mem_item, score in results:
                        if mem_item.id not in seen_ids:
                            seen_ids.add(mem_item.id)
                            retrieved_context.append(mem_item.to_dict())
            
            # 4. Add context_memory to item
            item["context_memory"] = retrieved_context
            
            # Write to file (jsonl format usually preferred for appending)
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            f_out.flush()
            
            count += 1
            print(f"Processed {sample_id}. Context memory size: {len(retrieved_context)}")

if __name__ == "__main__":
    # Configuration
    SOURCE_FILE = "/home/guozl/project/MemRL/Memory-CookBook/scripts/processed_locomo.json"
    TARGET_FILE = "/home/guozl/project/MemRL/Memory-CookBook/scripts/training_data_with_context.jsonl"
    K = 8
    MAX_ITEMS = 1024 # Process first 10 for testing as per user request "N items"
    
    process_data(SOURCE_FILE, TARGET_FILE, K, MAX_ITEMS)
