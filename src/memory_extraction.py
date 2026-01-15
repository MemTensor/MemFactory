# =============================================================================
# 记忆抽取模块 - Memory Extraction Module
# 本代码用于书籍《Memory Engineering》第10章配套代码
# 基于ReAct范式的智能记忆抽取Agent
# =============================================================================

"""
## Objective / 目标
- 从对话中抽取结构化记忆
- 支持多种抽取策略：简单抽取、ReAct Agent抽取

## I/O Spec / 输入输出约定
| Name | Type | Description |
|------|------|-------------|
| input | List[ConversationMessage] | 对话消息列表 |
| config | ExtractionConfig | 抽取配置 |
| result | ExtractionResult | 抽取结果（记忆列表+摘要） |
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
import json

try:
    from .common import (
        MemoryItem, ConversationMessage, ExtractionResult,
        MemoryType, UpdateAction,
        get_llm_client, get_embedding_client, get_neo4j_client, get_milvus_client,
        generate_id, current_timestamp, format_conversation
    )
except ImportError:
    from common import (
        MemoryItem, ConversationMessage, ExtractionResult,
        MemoryType, UpdateAction,
        get_llm_client, get_embedding_client, get_neo4j_client, get_milvus_client,
        generate_id, current_timestamp, format_conversation
    )


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class ExtractionConfig:
    """记忆抽取配置"""
    # 抽取策略
    strategy: Literal["simple", "react"] = "simple"
    
    # ReAct Agent参数
    max_steps: int = 3  # 最大思考步数
    
    # 存储配置
    auto_save: bool = True  # 是否自动保存到数据库
    auto_embed: bool = True  # 是否自动生成embedding
    
    # 调试参数
    verbose: bool = False
    
    # 用户信息
    user_id: str = "default_user"
    session_id: str = "default_session"


# =============================================================================
# Prompt模板
# =============================================================================

EXTRACTION_PROMPT_ZH = """您是记忆提取专家。
您的任务是根据用户与助手之间的对话，从用户的角度提取记忆。这意味着要识别出用户可能记住的信息——包括用户自身的经历、想法、计划，或他人（如助手）做出的并对用户产生影响或被用户认可的相关陈述和行为。

请执行以下操作：
1. 识别反映用户经历、信念、关切、决策、计划或反应的信息——包括用户认可或回应的来自助手的有意义信息。

2. 清晰解析所有时间、人物和事件的指代：
   - 如果可能，使用消息时间戳将相对时间表达（如"昨天"、"下周五"）转换为绝对日期。
   - 明确区分事件时间和消息时间。
   - 若提及具体地点，请包含在内。
   - 将所有代词、别名和模糊指代解析为全名或明确身份。

3. 始终以第三人称视角撰写，使用"用户"来指代用户，而不是使用第一人称。

4. 不要遗漏用户可能记住的任何信息。
   - 包括所有关键经历、想法、情绪反应和计划。
   - 优先考虑完整性和保真度，而非简洁性。

返回一个有效的JSON对象，结构如下：

{{
  "memory_list": [
    {{
      "key": "<字符串，唯一且简洁的记忆标题>",
      "memory_type": "<字符串，'LongTermMemory' 或 'UserMemory'>",
      "value": "<详细、独立且无歧义的记忆陈述>",
      "tags": ["<相关主题关键词列表>"]
    }}
  ],
  "summary": "<从用户视角自然总结上述记忆的段落，120-200字>"
}}

对话：
{conversation}

您的输出："""


REACT_SYSTEM_PROMPT = """你是一个基于ReAct范式的记忆抽取Agent。
你需要通过"思考-行动"循环来完成记忆抽取任务。

可用工具：
1. SearchContext - 检索历史记忆以补充上下文（用于解析指代）
2. UpdateBuffer - 将不完整信息暂存到缓冲区
3. CommitMemory - 提交完整的记忆抽取结果
4. Ignore - 判定为闲聊，不抽取记忆

输出格式（JSON）：
{{
  "thought": "你的思考过程",
  "action": "SearchContext/UpdateBuffer/CommitMemory/Ignore",
  "action_params": {{
    // SearchContext: {{"query": "检索查询"}}
    // UpdateBuffer: {{"fragment": "待暂存的片段"}}
    // CommitMemory: {{"memory_list": [...], "summary": "..."}}
    // Ignore: {{}}
  }}
}}

只输出JSON。"""


# =============================================================================
# 简单抽取器
# =============================================================================

class SimpleExtractor:
    """
    简单记忆抽取器
    直接使用LLM进行一次性抽取
    """
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.llm = get_llm_client()
    
    def extract(self, messages: List[ConversationMessage]) -> ExtractionResult:
        """
        从对话中抽取记忆
        
        Args:
            messages: 对话消息列表
            
        Returns:
            抽取结果
        """
        # 格式化对话
        conversation = format_conversation(messages)
        
        # 构建prompt
        prompt = EXTRACTION_PROMPT_ZH.format(conversation=conversation)
        
        # 调用LLM
        response = self.llm.chat(
            system_prompt="你是一个专业的记忆抽取专家。只输出JSON格式的结果。",
            user_prompt=prompt
        )
        
        # 解析结果
        result = self.llm.parse_json(response)
        
        if not result:
            # 解析失败，返回空结果
            return ExtractionResult(
                memory_list=[],
                summary="抽取失败",
                status="FAILED"
            )
        
        # 转换为MemoryItem列表
        memory_list = []
        for item in result.get("memory_list", []):
            memory = MemoryItem(
                id=generate_id(),
                key=item.get("key", ""),
                value=item.get("value", ""),
                memory_type=item.get("memory_type", "UserMemory"),
                tags=item.get("tags", []),
                user_id=self.config.user_id,
                session_id=self.config.session_id
            )
            memory_list.append(memory)
        
        if self.config.verbose:
            print(f"[SimpleExtractor] 抽取了 {len(memory_list)} 条记忆")
        
        return ExtractionResult(
            memory_list=memory_list,
            summary=result.get("summary", ""),
            status="SUCCESS"
        )


# =============================================================================
# ReAct Agent 抽取器
# =============================================================================

class MemoryBuffer:
    """记忆缓冲区：用于存储跨轮次的碎片信息"""
    
    def __init__(self):
        self._buffer: List[str] = []
    
    def append(self, fragment: str):
        """添加片段"""
        self._buffer.append(fragment)
    
    def get_contents(self) -> List[str]:
        """获取所有内容"""
        return self._buffer.copy()
    
    def clear(self):
        """清空缓冲区"""
        self._buffer.clear()
    
    def is_empty(self) -> bool:
        """是否为空"""
        return len(self._buffer) == 0


class ReActExtractor:
    """
    基于ReAct范式的记忆抽取Agent
    可以在生成最终记忆之前，自主决定是否需要查阅历史数据库以补充上下文
    """
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.llm = get_llm_client()
        self.embedding = get_embedding_client()
        self.neo4j = get_neo4j_client()
        self.milvus = get_milvus_client()
        self.buffer = MemoryBuffer()
    
    def extract(self, messages: List[ConversationMessage]) -> ExtractionResult:
        """
        ReAct主循环：Observation -> Thought -> Action -> Observation
        
        Args:
            messages: 对话消息列表
            
        Returns:
            抽取结果
        """
        # 初始化上下文
        conversation = format_conversation(messages)
        context = {
            "current": conversation,
            "history": messages[-5:] if len(messages) > 5 else messages,
            "buffer": self.buffer.get_contents(),
            "rag_context": []
        }
        
        thought_trace = []  # 记录思考轨迹
        
        for step in range(self.config.max_steps):
            if self.config.verbose:
                print(f"\n[ReActExtractor] Step {step + 1}/{self.config.max_steps}")
            
            # 1. 思考阶段 (Reasoning)
            agent_decision = self._think(context, thought_trace)
            
            if not agent_decision:
                continue
            
            # 2. 行动阶段 (Acting)
            action = agent_decision.get("action", "Ignore")
            reasoning = agent_decision.get("thought", "")
            params = agent_decision.get("action_params", {})
            
            thought_trace.append(f"Thought: {reasoning}")
            
            if self.config.verbose:
                print(f"  Thought: {reasoning[:100]}...")
                print(f"  Action: {action}")
            
            # --- 工具分发逻辑 ---
            
            # [Action 1] 检索增强：指代不清或需要验证事实时触发
            if action == "SearchContext":
                query = params.get("query", "")
                search_results = self._search_context(query)
                context["rag_context"] = search_results
                thought_trace.append(f"Observation: Found {len(search_results)} related memories")
                continue
            
            # [Action 2] 缓冲挂起：信息不全，等待后续补充
            elif action == "UpdateBuffer":
                fragment = params.get("fragment", "")
                self.buffer.append(fragment)
                return ExtractionResult(
                    memory_list=[],
                    summary="信息不完整，已暂存到缓冲区",
                    status="BUFFERED"
                )
            
            # [Action 3] 提交记忆：信息完整且有价值
            elif action == "CommitMemory":
                memory_list = self._parse_memories(params)
                summary = params.get("summary", "")
                self.buffer.clear()
                return ExtractionResult(
                    memory_list=memory_list,
                    summary=summary,
                    status="SUCCESS"
                )
            
            # [Action 4] 忽略：判定为闲聊
            elif action == "Ignore":
                return ExtractionResult(
                    memory_list=[],
                    summary="判定为闲聊，无需抽取记忆",
                    status="IGNORED"
                )
        
        # 兜底：超过最大步数，强制结束
        return ExtractionResult(
            memory_list=[],
            summary="超时",
            status="TIMEOUT"
        )
    
    def _think(self, context: Dict, trace: List[str]) -> Optional[Dict]:
        """Agent思考：根据当前上下文决定下一步行动"""
        user_prompt = f"""当前对话：
{context['current']}

缓冲区内容：{context['buffer']}

历史检索结果：{context['rag_context']}

思考轨迹：{trace}

请决定下一步行动。"""
        
        response = self.llm.chat(
            system_prompt=REACT_SYSTEM_PROMPT,
            user_prompt=user_prompt
        )
        
        return self.llm.parse_json(response)
    
    def _search_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索历史记忆"""
        # 生成查询向量
        query_emb = self.embedding.embed(query)
        
        # 向量检索
        results = self.milvus.search(query_emb, top_k=top_k)
        
        # 获取记忆详情
        memories = []
        for memory_id, score in results:
            mem = self.neo4j.get_memory(memory_id)
            if mem:
                memories.append({
                    "key": mem.key,
                    "value": mem.value,
                    "score": score
                })
        
        return memories
    
    def _parse_memories(self, params: Dict) -> List[MemoryItem]:
        """解析记忆参数为MemoryItem列表"""
        memory_list = []
        for item in params.get("memory_list", []):
            memory = MemoryItem(
                id=generate_id(),
                key=item.get("key", ""),
                value=item.get("value", ""),
                memory_type=item.get("memory_type", "UserMemory"),
                tags=item.get("tags", []),
                user_id=self.config.user_id,
                session_id=self.config.session_id
            )
            memory_list.append(memory)
        return memory_list


# =============================================================================
# 记忆抽取主类
# =============================================================================

class MemoryExtractor:
    """
    记忆抽取器：整合所有抽取策略的主入口
    
    使用方式：
        extractor = MemoryExtractor(config)
        result = extractor.run(messages)
    """
    
    def __init__(self, config: ExtractionConfig = None):
        self.config = config or ExtractionConfig()
        
        # 根据策略选择抽取器
        if self.config.strategy == "react":
            self._extractor = ReActExtractor(self.config)
        else:
            self._extractor = SimpleExtractor(self.config)
        
        # 数据库客户端
        self.neo4j = get_neo4j_client()
        self.milvus = get_milvus_client()
        self.embedding = get_embedding_client()
        
        if self.config.verbose:
            print(f"[MemoryExtractor] 初始化完成，策略: {self.config.strategy}")
    
    def run(self, messages: List[ConversationMessage]) -> ExtractionResult:
        """
        执行记忆抽取
        
        Args:
            messages: 对话消息列表
            
        Returns:
            抽取结果
        """
        # 1. 抽取记忆
        result = self._extractor.extract(messages)
        
        # 2. 自动保存
        if self.config.auto_save and result.status == "SUCCESS":
            self._save_memories(result.memory_list)
        
        return result
    
    def _save_memories(self, memories: List[MemoryItem]):
        """保存记忆到数据库"""
        for memory in memories:
            # 生成embedding
            if self.config.auto_embed:
                text = f"{memory.key} {memory.value}"
                memory.embedding = self.embedding.embed(text)
                
                # 保存到Milvus
                self.milvus.insert(memory.id, memory.embedding)
            
            # 保存到Neo4j
            self.neo4j.save_memory(memory)
            
            if self.config.verbose:
                print(f"[MemoryExtractor] 保存记忆: {memory.id} - {memory.key}")


# =============================================================================
# 便捷函数
# =============================================================================

def extract_memories(
    messages: List[ConversationMessage],
    strategy: str = "simple",
    auto_save: bool = True,
    verbose: bool = False,
    user_id: str = "default_user"
) -> ExtractionResult:
    """
    便捷函数：从对话中抽取记忆
    
    Args:
        messages: 对话消息列表
        strategy: 抽取策略 ("simple" 或 "react")
        auto_save: 是否自动保存
        verbose: 是否打印调试信息
        user_id: 用户ID
        
    Returns:
        抽取结果
    """
    config = ExtractionConfig(
        strategy=strategy,
        auto_save=auto_save,
        verbose=verbose,
        user_id=user_id
    )
    extractor = MemoryExtractor(config)
    return extractor.run(messages)


# =============================================================================
# 示例代码
# =============================================================================

def example_extraction():
    """记忆抽取示例"""
    print("=" * 60)
    print("记忆抽取示例")
    print("=" * 60)
    
    # 构造示例对话
    messages = [
        ConversationMessage(
            role="user",
            content="嗨Jerry！昨天下午3点我和团队开了个会，讨论新项目。",
            timestamp="2025-06-26T15:00:00"
        ),
        ConversationMessage(
            role="assistant",
            content="哦Tom！你觉得团队能在12月15日前完成吗？",
            timestamp="2025-06-26T15:01:00"
        ),
        ConversationMessage(
            role="user",
            content="我有点担心。后端要到12月10日才能完成，所以测试时间会很紧。",
            timestamp="2025-06-26T15:02:00"
        ),
        ConversationMessage(
            role="assistant",
            content="也许提议延期？",
            timestamp="2025-06-26T15:03:00"
        ),
        ConversationMessage(
            role="user",
            content="好主意。我明天上午9:30的会上提一下——也许把截止日期推迟到1月5日。",
            timestamp="2025-06-26T16:21:00"
        )
    ]
    
    # 使用简单策略抽取
    config = ExtractionConfig(
        strategy="simple",
        auto_save=False,  # 示例中不保存
        verbose=True,
        user_id="tom"
    )
    
    extractor = MemoryExtractor(config)
    result = extractor.run(messages)
    
    print(f"\n状态: {result.status}")
    print(f"摘要: {result.summary}")
    print(f"\n抽取的记忆 ({len(result.memory_list)} 条):")
    for mem in result.memory_list:
        print(f"  - [{mem.memory_type}] {mem.key}")
        print(f"    {mem.value[:80]}...")
        print(f"    标签: {mem.tags}")
    
    return result


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    example_extraction()
