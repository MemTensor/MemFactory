# =============================================================================
# 记忆检索模块 - Memory Search Module
# 本代码用于书籍《Memory Engineering》第12章配套代码
# 涵盖：被动检索、主动检索、DTR自适应检索、上下文注入
# =============================================================================

"""
## Objective / 目标
- 实现多种记忆检索策略
- 支持被动检索和主动检索
- 支持DTR（Decide-Then-Retrieve）自适应检索
- 实现上下文注入策略

## I/O Spec / 输入输出约定
| Name | Type | Description |
|------|------|-------------|
| input | str / Query | 检索查询 |
| config | SearchConfig | 检索配置 |
| result | SearchResult | 检索结果（记忆列表+相关性分数） |
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Literal
from datetime import datetime, timedelta
import math

try:
    from .common import (
        MemoryItem, SearchResult, ConversationMessage,
        MemoryStatus,
        get_llm_client, get_embedding_client, get_neo4j_client, get_milvus_client,
        generate_id, current_timestamp
    )
except ImportError:
    from common import (
        MemoryItem, SearchResult, ConversationMessage,
        MemoryStatus,
        get_llm_client, get_embedding_client, get_neo4j_client, get_milvus_client,
        generate_id, current_timestamp
    )


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class SearchConfig:
    """记忆检索配置"""
    # 检索策略
    strategy: Literal["passive", "active", "dtr"] = "passive"
    
    # 检索参数
    top_k: int = 10                      # 返回数量
    similarity_threshold: float = 0.3    # 相似度阈值
    
    # DTR参数
    uncertainty_threshold: float = 0.5   # 不确定性阈值
    
    # 时间衰减参数
    use_time_decay: bool = True          # 是否使用时间衰减
    time_decay_lambda: float = 0.04      # 衰减系数（每天）
    
    # 注入策略
    injection_mode: Literal["prefix", "suffix", "partition"] = "partition"
    injection_format: Literal["summary", "slots", "constraints"] = "summary"
    max_inject_chars: int = 2000         # 最大注入字符数
    
    # 用户信息
    user_id: str = "default_user"
    
    # 调试参数
    verbose: bool = False


@dataclass
class Query:
    """检索查询"""
    text: str                            # 查询文本
    context: Optional[str] = None        # 上下文
    timestamp: str = ""                  # 时间戳
    intent: Optional[str] = None         # 意图（可选）
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = current_timestamp()


# =============================================================================
# 被动检索器
# =============================================================================

class PassiveRetriever:
    """
    被动检索器：响应式的信息补全机制
    当外部请求或当前推理显式暴露出信息缺口时，立即调用长期记忆
    """
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.embedding = get_embedding_client()
        self.neo4j = get_neo4j_client()
        self.milvus = get_milvus_client()
    
    def search(self, query: Query) -> SearchResult:
        """
        执行被动检索
        
        Args:
            query: 检索查询
            
        Returns:
            检索结果
        """
        # 1. 生成查询向量
        query_emb = self.embedding.embed(query.text)
        
        # 2. 向量检索
        vector_results = self.milvus.search(query_emb, top_k=self.config.top_k * 2)
        
        # 3. 获取记忆详情并计算最终分数
        scored_memories = []
        now = datetime.now()
        
        for memory_id, vec_score in vector_results:
            memory = self.neo4j.get_memory(memory_id)
            if not memory:
                continue
            
            # 过滤非活跃记忆
            if memory.status != MemoryStatus.ACTIVATED.value:
                continue
            
            # 过滤用户
            if self.config.user_id and memory.user_id != self.config.user_id:
                if memory.user_id != "default_user":
                    continue
            
            # 计算最终分数
            final_score = vec_score
            
            # 时间衰减
            if self.config.use_time_decay:
                time_score = self._calculate_time_score(memory, now)
                final_score = 0.7 * vec_score + 0.3 * time_score
            
            if final_score >= self.config.similarity_threshold:
                scored_memories.append((memory, final_score))
        
        # 4. 排序并截取
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        scored_memories = scored_memories[:self.config.top_k]
        
        if self.config.verbose:
            print(f"[PassiveRetriever] 检索到 {len(scored_memories)} 条记忆")
        
        return SearchResult(
            memories=scored_memories,
            query=query.text,
            total_found=len(scored_memories)
        )
    
    def _calculate_time_score(self, memory: MemoryItem, now: datetime) -> float:
        """计算时间新鲜度分数"""
        try:
            mem_time = datetime.fromisoformat(memory.updated_at.replace('Z', ''))
            days = max((now - mem_time).days, 0)
            return math.exp(-self.config.time_decay_lambda * days)
        except:
            return 0.5


# =============================================================================
# 主动检索器
# =============================================================================

class ActiveRetriever:
    """
    主动检索器：预测式的记忆准备机制
    由系统内部状态变化、长期策略或预测性判断所触发
    """
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.embedding = get_embedding_client()
        self.neo4j = get_neo4j_client()
        self.milvus = get_milvus_client()
        self.llm = get_llm_client()
    
    def search(self, query: Query, 
               related_topics: List[str] = None) -> SearchResult:
        """
        执行主动检索
        
        Args:
            query: 检索查询
            related_topics: 相关主题（用于扩展检索）
            
        Returns:
            检索结果
        """
        all_memories = []
        
        # 1. 基础检索
        query_emb = self.embedding.embed(query.text)
        base_results = self.milvus.search(query_emb, top_k=self.config.top_k)
        
        for memory_id, score in base_results:
            memory = self.neo4j.get_memory(memory_id)
            if memory and memory.status == MemoryStatus.ACTIVATED.value:
                all_memories.append((memory, score))
        
        # 2. 扩展检索：相关主题
        if related_topics:
            for topic in related_topics:
                topic_emb = self.embedding.embed(topic)
                topic_results = self.milvus.search(topic_emb, top_k=3)
                
                for memory_id, score in topic_results:
                    memory = self.neo4j.get_memory(memory_id)
                    if memory and memory.status == MemoryStatus.ACTIVATED.value:
                        # 降低扩展检索的分数
                        all_memories.append((memory, score * 0.8))
        
        # 3. 去重并排序
        seen_ids = set()
        unique_memories = []
        for mem, score in sorted(all_memories, key=lambda x: x[1], reverse=True):
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                unique_memories.append((mem, score))
        
        unique_memories = unique_memories[:self.config.top_k]
        
        if self.config.verbose:
            print(f"[ActiveRetriever] 检索到 {len(unique_memories)} 条记忆")
        
        return SearchResult(
            memories=unique_memories,
            query=query.text,
            total_found=len(unique_memories)
        )
    
    def proactive_recall(self, context: str) -> SearchResult:
        """
        主动回忆：基于当前上下文预测可能需要的记忆
        
        Args:
            context: 当前上下文
            
        Returns:
            检索结果
        """
        # 使用LLM分析上下文，提取可能需要的记忆主题
        prompt = f"""Analyze the following context and extract topics of historical information that may need to be recalled.

Context:
{context}

Output JSON format:
{{"topics": ["Topic 1", "Topic 2", ...], "reason": "Analysis reason"}}

Output JSON only."""
        
        response = self.llm.chat(
            system_prompt="You are a memory analysis expert.",
            user_prompt=prompt
        )
        
        result = self.llm.parse_json(response)
        if not result:
            return SearchResult(memories=[], query=context, total_found=0)
        
        topics = result.get("topics", [])
        
        if self.config.verbose:
            print(f"[ActiveRetriever] Identified topics: {topics}")
        
        # 基于主题进行检索
        query = Query(text=context)
        return self.search(query, related_topics=topics)


# =============================================================================
# DTR自适应检索器
# =============================================================================

class DTRRetriever:
    """
    DTR（Decide-Then-Retrieve）自适应检索器
    先决策、再检索：当系统判断外部信息确实能带来收益时才触发检索
    """
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.embedding = get_embedding_client()
        self.neo4j = get_neo4j_client()
        self.milvus = get_milvus_client()
        self.llm = get_llm_client()
    
    def search(self, query: Query) -> SearchResult:
        """
        执行DTR检索
        
        Args:
            query: 检索查询
            
        Returns:
            检索结果
        """
        # 1. 决策阶段：判断是否需要检索
        should_retrieve, uncertainty = self._decide_retrieval(query)
        
        if self.config.verbose:
            print(f"[DTRRetriever] Uncertainty: {uncertainty:.3f}, Should retrieve: {should_retrieve}")
        
        if not should_retrieve:
            return SearchResult(
                memories=[],
                query=query.text,
                total_found=0
            )
        
        # 2. 双路径检索
        results = self._dual_path_search(query)
        
        # 3. 自适应信息选择（AIS）
        selected = self._adaptive_selection(query, results)
        
        return SearchResult(
            memories=selected,
            query=query.text,
            total_found=len(selected)
        )
    
    def _decide_retrieval(self, query: Query) -> Tuple[bool, float]:
        """
        决策是否需要检索
        基于生成不确定性判断
        
        Returns:
            (是否检索, 不确定性分数)
        """
        # 使用LLM生成草稿答案并估计不确定性
        prompt = f"""Generate a brief draft answer to the following question and evaluate your level of certainty.

Question: {query.text}

Output JSON format:
{{"draft_answer": "Draft answer", "confidence": 0.0-1.0, "reason": "Reason for certainty level"}}

Output JSON only."""
        
        response = self.llm.chat(
            system_prompt="You are a knowledge assessment expert.",
            user_prompt=prompt
        )
        
        result = self.llm.parse_json(response)
        if not result:
            # 解析失败，默认检索
            return True, 0.5
        
        confidence = result.get("confidence", 0.5)
        uncertainty = 1.0 - confidence
        
        # 不确定性超过阈值则检索
        should_retrieve = uncertainty > self.config.uncertainty_threshold
        
        return should_retrieve, uncertainty
    
    def _dual_path_search(self, query: Query) -> List[Tuple[MemoryItem, float, str]]:
        """
        双路径检索：原始查询 + 伪上下文
        
        Returns:
            (记忆, 分数, 来源路径) 列表
        """
        results = []
        
        # 路径A：基于原始查询
        query_emb = self.embedding.embed(query.text)
        path_a_results = self.milvus.search(query_emb, top_k=self.config.top_k)
        
        for memory_id, score in path_a_results:
            memory = self.neo4j.get_memory(memory_id)
            if memory and memory.status == MemoryStatus.ACTIVATED.value:
                results.append((memory, score, "path_a"))
        
        # 路径B：生成伪上下文并检索
        pseudo_context = self._generate_pseudo_context(query)
        if pseudo_context:
            pseudo_emb = self.embedding.embed(pseudo_context)
            path_b_results = self.milvus.search(pseudo_emb, top_k=self.config.top_k)
            
            for memory_id, score in path_b_results:
                memory = self.neo4j.get_memory(memory_id)
                if memory and memory.status == MemoryStatus.ACTIVATED.value:
                    results.append((memory, score, "path_b"))
        
        return results
    
    def _generate_pseudo_context(self, query: Query) -> str:
        """生成伪上下文：补全用户意图"""
        prompt = f"""Expand the following user query into a more complete and searchable description.
Do not fabricate facts, just make the query more specific and clear.

User query: {query.text}

Output the expanded description (one paragraph):"""
        
        response = self.llm.chat(
            system_prompt="You are a query expansion expert.",
            user_prompt=prompt
        )
        
        return response.strip()
    
    def _adaptive_selection(self, query: Query, 
                           results: List[Tuple[MemoryItem, float, str]]) -> List[Tuple[MemoryItem, float]]:
        """
        自适应信息选择（AIS）
        为每个候选计算综合分数
        """
        query_emb = self.embedding.embed(query.text)
        
        scored = []
        seen_ids = set()
        now = datetime.now()
        
        for memory, vec_score, path in results:
            if memory.id in seen_ids:
                continue
            seen_ids.add(memory.id)
            
            # 计算与查询的相似度
            if memory.embedding:
                s1 = self.embedding.similarity(query_emb, memory.embedding)
            else:
                mem_emb = self.embedding.embed(f"{memory.key} {memory.value}")
                s1 = self.embedding.similarity(query_emb, mem_emb)
            
            # 时间新鲜度
            time_score = 0.5
            if self.config.use_time_decay:
                try:
                    mem_time = datetime.fromisoformat(memory.updated_at.replace('Z', ''))
                    days = max((now - mem_time).days, 0)
                    time_score = math.exp(-self.config.time_decay_lambda * days)
                except:
                    pass
            
            # 综合分数
            final_score = 0.6 * s1 + 0.2 * vec_score + 0.2 * time_score
            
            if final_score >= self.config.similarity_threshold:
                scored.append((memory, final_score))
        
        # 排序并返回
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.config.top_k]


# =============================================================================
# 上下文注入器
# =============================================================================

class ContextInjector:
    """
    上下文注入器：将检索结果注入到推理上下文中
    支持多种注入策略：前缀、后缀、分区
    支持多种呈现格式：摘要、槽位、约束
    """
    
    def __init__(self, config: SearchConfig):
        self.config = config
    
    def inject(self, query: str, 
               memories: List[Tuple[MemoryItem, float]]) -> str:
        """
        将记忆注入到上下文中
        
        Args:
            query: 用户查询
            memories: 检索到的记忆列表
            
        Returns:
            注入后的上下文
        """
        if not memories:
            return query
        
        # 1. 格式化记忆
        formatted = self._format_memories(memories)
        
        # 2. 截断到最大长度
        if len(formatted) > self.config.max_inject_chars:
            formatted = formatted[:self.config.max_inject_chars] + "..."
        
        # 3. 根据策略注入
        if self.config.injection_mode == "prefix":
            return f"{formatted}\n\n用户查询：{query}"
        elif self.config.injection_mode == "suffix":
            return f"用户查询：{query}\n\n相关记忆：\n{formatted}"
        else:  # partition
            return f"=== 记忆证据 ===\n{formatted}\n\n=== 用户查询 ===\n{query}"
    
    def _format_memories(self, memories: List[Tuple[MemoryItem, float]]) -> str:
        """根据格式策略格式化记忆"""
        if self.config.injection_format == "summary":
            return self._format_as_summary(memories)
        elif self.config.injection_format == "slots":
            return self._format_as_slots(memories)
        else:  # constraints
            return self._format_as_constraints(memories)
    
    def _format_as_summary(self, memories: List[Tuple[MemoryItem, float]]) -> str:
        """格式化为摘要形式"""
        lines = []
        for mem, score in memories:
            date = mem.updated_at[:10] if mem.updated_at else "未知"
            lines.append(f"- [{mem.memory_type}|{date}] {mem.value}")
        return "\n".join(lines)
    
    def _format_as_slots(self, memories: List[Tuple[MemoryItem, float]]) -> str:
        """格式化为槽位形式"""
        slots = {}
        for mem, score in memories:
            key = mem.key
            if key not in slots or score > slots[key][1]:
                slots[key] = (mem.value, score)
        
        lines = []
        for key, (value, score) in slots.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)
    
    def _format_as_constraints(self, memories: List[Tuple[MemoryItem, float]]) -> str:
        """格式化为约束形式"""
        lines = ["约束条件："]
        for mem, score in memories:
            lines.append(f"- [constraint:{mem.memory_type}] {mem.value}")
        return "\n".join(lines)


# =============================================================================
# 记忆检索主类
# =============================================================================

class MemorySearcher:
    """
    记忆检索器：整合所有检索策略的主入口
    
    使用方式：
        searcher = MemorySearcher(config)
        result = searcher.run(query)
    """
    
    def __init__(self, config: SearchConfig = None):
        self.config = config or SearchConfig()
        
        # 根据策略选择检索器
        if self.config.strategy == "active":
            self._retriever = ActiveRetriever(self.config)
        elif self.config.strategy == "dtr":
            self._retriever = DTRRetriever(self.config)
        else:
            self._retriever = PassiveRetriever(self.config)
        
        # 上下文注入器
        self.injector = ContextInjector(self.config)
        
        if self.config.verbose:
            print(f"[MemorySearcher] 初始化完成，策略: {self.config.strategy}")
    
    def run(self, query: str, context: str = None) -> SearchResult:
        """
        执行记忆检索
        
        Args:
            query: 查询文本
            context: 上下文（可选）
            
        Returns:
            检索结果
        """
        q = Query(text=query, context=context)
        return self._retriever.search(q)
    
    def search_and_inject(self, query: str, context: str = None) -> Tuple[SearchResult, str]:
        """
        检索并注入上下文
        
        Args:
            query: 查询文本
            context: 上下文（可选）
            
        Returns:
            (检索结果, 注入后的上下文)
        """
        result = self.run(query, context)
        injected = self.injector.inject(query, result.memories)
        return result, injected


# =============================================================================
# 便捷函数
# =============================================================================

def search_memories(
    query: str,
    strategy: str = "passive",
    top_k: int = 10,
    verbose: bool = False,
    user_id: str = "default_user"
) -> SearchResult:
    """
    便捷函数：检索记忆
    
    Args:
        query: 查询文本
        strategy: 检索策略 ("passive", "active", "dtr")
        top_k: 返回数量
        verbose: 是否打印调试信息
        user_id: 用户ID
        
    Returns:
        检索结果
    """
    config = SearchConfig(
        strategy=strategy,
        top_k=top_k,
        verbose=verbose,
        user_id=user_id
    )
    searcher = MemorySearcher(config)
    return searcher.run(query)


# =============================================================================
# 示例代码
# =============================================================================

def example_search():
    """记忆检索示例"""
    print("=" * 60)
    print("记忆检索示例")
    print("=" * 60)
    
    # 先添加一些测试记忆
    from .common import get_neo4j_client, get_milvus_client, get_embedding_client
    
    neo4j = get_neo4j_client()
    milvus = get_milvus_client()
    embedding = get_embedding_client()
    
    # 创建测试记忆
    test_memories = [
        MemoryItem(
            id=generate_id(),
            key="饮品偏好",
            value="用户喜欢喝美式咖啡，不加糖",
            memory_type="UserMemory",
            tags=["偏好", "饮品"]
        ),
        MemoryItem(
            id=generate_id(),
            key="工作信息",
            value="用户是一名软件工程师，在科技公司工作",
            memory_type="UserMemory",
            tags=["事实", "工作"]
        ),
        MemoryItem(
            id=generate_id(),
            key="项目进度",
            value="用户正在开发一个AI项目，预计下周完成",
            memory_type="LongTermMemory",
            tags=["项目", "AI"]
        )
    ]
    
    # 保存测试记忆
    for mem in test_memories:
        text = f"{mem.key} {mem.value}"
        mem.embedding = embedding.embed(text)
        neo4j.save_memory(mem)
        milvus.insert(mem.id, mem.embedding)
        print(f"  保存: {mem.key}")
    
    # 测试被动检索
    print("\n--- 被动检索 ---")
    config = SearchConfig(
        strategy="passive",
        top_k=5,
        verbose=True
    )
    searcher = MemorySearcher(config)
    result = searcher.run("用户喜欢喝什么饮料？")
    
    print(f"查询: {result.query}")
    print(f"找到: {result.total_found} 条记忆")
    for mem, score in result.memories:
        print(f"  [{score:.3f}] {mem.key}: {mem.value[:50]}...")
    
    # 测试DTR检索
    print("\n--- DTR检索 ---")
    config = SearchConfig(
        strategy="dtr",
        top_k=5,
        verbose=True
    )
    searcher = MemorySearcher(config)
    result = searcher.run("项目什么时候能完成？")
    
    print(f"查询: {result.query}")
    print(f"找到: {result.total_found} 条记忆")
    for mem, score in result.memories:
        print(f"  [{score:.3f}] {mem.key}: {mem.value[:50]}...")
    
    # 测试上下文注入
    print("\n--- 上下文注入 ---")
    result, injected = searcher.search_and_inject("用户的工作是什么？")
    print(f"注入后的上下文:\n{injected}")
    
    return result


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    example_search()
