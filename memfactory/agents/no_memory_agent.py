import torch
from typing import Any, Dict, List, Optional

from ..common.registry import AGENT_REGISTRY
from .base import BaseAgent
from ..modules.base import Samples


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


@AGENT_REGISTRY.register("no_memory")
class NoMemoryAgent(BaseAgent):
    """
    A vanilla GRPO baseline with no recurrent memory state and no memory
    extraction/update/retrieval operation. It directly optimizes final answers
    from the question and available context.
    """

    def __init__(self, tokenizer, device="cuda", **kwargs):
        super().__init__(tokenizer, device)
        self.num_generations = kwargs.get("num_generations", 16)
        self.max_generate_length = kwargs.get("max_generate_length", 2048)
        self.max_prompt_length = kwargs.get("max_prompt_length", 4096)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _generate_with_pytorch(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_generate_length,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
            )
        return outputs

    def _build_prompt(self, question: str, context_ids: List[int]) -> str:
        # Leave room for the question and chat template. This baseline is meant
        # to avoid any learned memory state, not to guarantee full-context access.
        max_context_tokens = max(1, self.max_prompt_length - 512)
        truncated_context_ids = context_ids[:max_context_tokens]
        context_text = self.tokenizer.decode(truncated_context_ids, skip_special_tokens=True)
        return NO_MEMORY_QA_TEMPLATE.format(prompt=question, context=context_text)

    def _to_samples(self, results: List[tuple[str, str, float]]) -> Samples:
        prompts_ids = [self.tokenizer.encode(p, add_special_tokens=False) for p, _, _ in results]
        responses_ids = [
            self.tokenizer.encode(r, add_special_tokens=False) + [self.tokenizer.eos_token_id]
            for _, r, _ in results
        ]
        advantages = [a for _, _, a in results]

        max_p_len = max(len(ids) for ids in prompts_ids)
        padded_prompts = []
        prompt_masks = []
        for p_ids in prompts_ids:
            pad_len = max_p_len - len(p_ids)
            padded_prompts.append([self.tokenizer.pad_token_id] * pad_len + p_ids)
            prompt_masks.append([0] * pad_len + [1] * len(p_ids))

        max_r_len = max(len(ids) for ids in responses_ids)
        padded_responses = []
        response_masks = []
        response_att_masks = []
        for r_ids in responses_ids:
            pad_len = max_r_len - len(r_ids)
            padded_responses.append(r_ids + [self.tokenizer.pad_token_id] * pad_len)
            response_masks.append([1] * len(r_ids) + [0] * pad_len)
            response_att_masks.append([1] * len(r_ids) + [0] * pad_len)

        input_ids = torch.tensor(
            [p + r for p, r in zip(padded_prompts, padded_responses)],
            device=self.device,
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [p + r for p, r in zip(prompt_masks, response_att_masks)],
            device=self.device,
            dtype=torch.long,
        )
        action_mask = torch.tensor(response_masks, device=self.device, dtype=torch.bool)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        return Samples(
            prompt_response_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=max_r_len,
            rewards=advantages_tensor,
            prompt_length=torch.tensor([max_p_len] * len(results), device=self.device),
            response_length=action_mask.sum(dim=1),
            step_type="no_memory",
        )

    def rollout(self, model: Any, batch_data: Dict[str, Any], **kwargs) -> Optional[Samples]:
        reward_fn = kwargs.get("reward_fn")
        if not reward_fn:
            raise ValueError("reward_fn is required for rollout")

        results = []
        bs = len(batch_data["context_ids"])

        for i in range(bs):
            context_ids = batch_data["context_ids"][i]
            question = batch_data["question"][i]
            ground_truth = batch_data["ground_truth"][i]

            prompt = self._build_prompt(question, context_ids)
            step_prompts = [prompt] * self.num_generations
            msgs_list = [[{"role": "user", "content": p}] for p in step_prompts]
            formatted_prompts = [
                self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                for m in msgs_list
            ]

            tokenized = self.tokenizer(
                formatted_prompts,
                padding=True,
                truncation=True,
                max_length=self.max_prompt_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self._generate_with_pytorch(model, tokenized["input_ids"], tokenized["attention_mask"])
            generated_texts = self.tokenizer.batch_decode(
                outputs[:, tokenized["input_ids"].size(1):],
                skip_special_tokens=True,
            )

            scores = reward_fn(
                generated_texts,
                [ground_truth] * self.num_generations,
                [question] * self.num_generations,
            )
            scores_tensor = torch.tensor(scores, dtype=torch.float32, device=self.device)
            mean_score = scores_tensor.mean()
            std_score = scores_tensor.std()

            if std_score.item() < 1e-6:
                continue

            advantages = (scores_tensor - mean_score) / (std_score + 1e-8)
            for j in range(self.num_generations):
                results.append((formatted_prompts[j], generated_texts[j], advantages[j].item()))

        if not results:
            return None
        return self._to_samples(results)

    def inference(self, batch_data: Dict[str, Any], **kwargs) -> List[str]:
        return []
