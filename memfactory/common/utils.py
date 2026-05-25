import json
import os
import sys
import torch
import json
from typing import List, Optional, Union, Any, Dict
from concurrent.futures import ThreadPoolExecutor

# =============================================================================
# LLM service
# =============================================================================
from openai import OpenAI
try:
    from dotenv import load_dotenv
    # Try loading .env from common project locations.
    for env_path in ['.env', '../.env', '../../.env']:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            break
except ImportError:
    print("Warning: dotenv is unavailable; OpenAI-based services may not work.")
    pass  

# OpenAI LLM API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-nano")
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "300"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))


class LLMClient:
    """
    LLM client wrapper around OpenAI-compatible chat APIs.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=OPENAI_TIMEOUT_SECONDS,
            max_retries=OPENAI_MAX_RETRIES,
        )
        self.model = LLM_MODEL
        self._initialized = True
        print(f"[LLMClient] Initialized with model: {self.model}")
    
    def chat(self, system_prompt: str, user_prompt: str, 
             temperature: float = 0.3) -> str:
        """
        Call the LLM for a chat-style completion.
        
        Args:
            system_prompt: System prompt.
            user_prompt: User input.
            temperature: Sampling temperature.
            
        Returns:
            LLM response text.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLMClient] API call failed: {e}")
            return ""
    
    def parse_json(self, response: str) -> Optional[Dict]:
        """Parse a JSON response."""
        try:
            # Try to extract a fenced JSON block.
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            # Trim surrounding whitespace.
            response = response.strip()
            # Strip chain-of-thought wrappers if present.
            if response.startswith("<think>"):
                response = response.split("</think>")[-1]
                response = response.strip()
            # Try to parse a JSON object after handling possible prefixes.
            if not response.startswith("{"):
                print("[LLMClient-parse_json] Response does not start with '{'; cannot parse", response[:100])

            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"[LLMClient] JSON parse failed: {e}")
            return None


TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

TEMPLATE_FINAL_BOXED = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

NO_MEMORY_QA_TEMPLATE = """You are presented with a problem and a context that may contain the answer.
Answer the problem based on the context and put the answer in \\boxed{{}}.

<problem>
{prompt}
</problem>

<context>
{context}
</context>

Your answer:
"""

def truncate_context_ids(context_ids: List[int], max_tokens: int, truncation_side: str = "head") -> List[int]:
    if max_tokens <= 0:
        return []
    if len(context_ids) <= max_tokens:
        return list(context_ids)
    if truncation_side == "head":
        return list(context_ids[:max_tokens])
    if truncation_side == "tail":
        return list(context_ids[-max_tokens:])
    if truncation_side == "middle":
        head_tokens = max_tokens // 2
        tail_tokens = max_tokens - head_tokens
        return list(context_ids[:head_tokens] + context_ids[-tail_tokens:])
    raise ValueError(f"Unsupported context_truncation_side: {truncation_side}")

def build_no_memory_qa_prompt(
    tokenizer,
    question: str,
    context_ids: List[int],
    max_prompt_length: Optional[int] = None,
    context_truncation_side: str = "head",
    apply_chat_template: bool = True,
) -> str:
    """
    Build the direct no-memory QA prompt while preserving the answer instruction.
    The context is truncated before template rendering so tokenizer truncation does
    not accidentally remove the tail of the prompt.
    """
    context_ids = list(context_ids or [])
    context_budget = len(context_ids)

    if max_prompt_length is not None and max_prompt_length > 0:
        empty_prompt = NO_MEMORY_QA_TEMPLATE.format(prompt=question, context="")
        if apply_chat_template:
            empty_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": empty_prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        overhead = len(tokenizer.encode(empty_prompt, add_special_tokens=False))
        context_budget = max(0, max_prompt_length - overhead - 8)

    truncated_context_ids = truncate_context_ids(context_ids, context_budget, context_truncation_side)

    for _ in range(4):
        context_text = tokenizer.decode(truncated_context_ids, skip_special_tokens=True)
        prompt = NO_MEMORY_QA_TEMPLATE.format(prompt=question, context=context_text)
        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        if max_prompt_length is None or max_prompt_length <= 0:
            return prompt

        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        if prompt_len <= max_prompt_length or context_budget <= 0:
            return prompt

        overflow = prompt_len - max_prompt_length
        context_budget = max(0, context_budget - overflow - 8)
        truncated_context_ids = truncate_context_ids(context_ids, context_budget, context_truncation_side)

    return prompt

JUDGE_PROMPT = """Please judge whether the predicted answer is correct based on the standard answer.

Question: {question}
Standard Answer: {answer}
Predicted Answer: {prediction}

Is the predicted answer consistent with the standard answer? Please output only "True" or "False".
"""

def extract_boxed_content(text):
    """
    Extracts the content inside the last \boxed{...} in the text.
    Handles nested braces correctly.
    """
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    
    i = idx + 6
    while i < len(text) and text[i].isspace():
        i += 1
        
    if i >= len(text) or text[i] != '{':
        return None
    
    start_brace = i
    balance = 0
    content_start = start_brace + 1
    
    for j in range(start_brace, len(text)):
        if text[j] == '{':
            balance += 1
        elif text[j] == '}':
            balance -= 1
            if balance == 0:
                return text[content_start:j]
                
    return None

def evaluate_memory_agent(response, ground_truth, question="", llm_client=None):
    try:
        boxed_content = extract_boxed_content(response)
        if boxed_content is None:
            return 0.0
            
        assert llm_client is not None, "llm_client must not be None"
        llm = llm_client
        judge_prompt = JUDGE_PROMPT.format(
            question=question, 
            answer=ground_truth, 
            prediction=boxed_content
        )
        
        judge_result = llm.chat("You are an impartial judge.", judge_prompt)
        
        if "<think>" in judge_result:
            if "</think>" in judge_result:
                judge_result = judge_result.split("</think>")[-1].strip()
            else:
                judge_result = judge_result[-100:].strip()
                
        if "True" in judge_result:
            return 1.0
        return 0.0
    except Exception as e:
        print(f"Evaluation Error: {e}")
        return 0.0

def evaluate_memory_agent_batch(responses, ground_truths, questions, max_workers=16, llm_client=None):
    if llm_client is None:
        assert False, "llm_client should not be None"
        # llm_client = LLMClient() optional
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(len(responses)):
            futures.append(
                executor.submit(
                    evaluate_memory_agent, 
                    responses[i], 
                    ground_truths[i] if isinstance(ground_truths, list) else ground_truths,
                    questions[i] if isinstance(questions, list) else questions,
                    llm_client
                )
            )
        scores = [f.result() for f in futures]
    return scores


def parse_json_from_text(response: str) -> Optional[Dict]:
        """Parse a JSON response."""
        try:
            # Try to extract a fenced JSON block.
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            # Trim surrounding whitespace.
            response = response.strip()
            # Strip chain-of-thought wrappers if present.
            if response.startswith("<think>"):
                response = response.split("</think>")[-1]
                response = response.strip()
            # Try to parse a JSON object after handling possible prefixes.
            if not response.startswith("{"):
                print("Extraction result does not start with '{'; cannot parse", response[:100])

            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Extraction result JSON parse failed: {e}")
            return {}
