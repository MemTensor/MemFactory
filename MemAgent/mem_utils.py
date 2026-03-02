
TEMPLATE = """You are a memory assistant. 
Prompt: {prompt}
Memory: {memory}
Current Chunk: {chunk}
Please update the memory based on the current chunk."""

TEMPLATE_FINAL_BOXED = """Based on the memory, answer the prompt.
Prompt: {prompt}
Memory: {memory}
Output the final answer boxed in \\boxed{{}}."""

def evaluate_memory_agent(response, ground_truth):
    """
    Placeholder for evaluation.
    Returns a scalar score.
    """
    # Simple dummy evaluation for now
    return 0.5
