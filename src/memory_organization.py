# =============================================================================
# 记忆组织模块 - Memory Organization Module
# 本代码用于书籍《Memory Engineering》第11章配套代码
# 涵盖：时间结构、事件结构、语义结构、层级抽象、关系建模
# =============================================================================

"""
## Objective / 目标
- 将零散的记忆节点组织成可推理的结构化网络
- 支持时间结构、事件结构、语义结构、层级抽象

## I/O Spec / 输入输出约定
| Name | Type | Description |
|------|------|-------------|
| input | List[MemoryItem] | 记忆列表 |
| config | OrganizationConfig | 组织配置 |
| result | MemoryGraph | 组织后的记忆图谱 |
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set, Literal
from datetime import datetime, timedelta
from collections import defaultdict
import uuid

try:
    from .common import (
        MemoryItem, Edge, RelationType,
        get_llm_client, get_embedding_client, get_neo4j_client,
        generate_id, current_timestamp
    )
except ImportError:
    from common import (
        MemoryItem, Edge, RelationType,
        get_llm_client, get_embedding_client, get_neo4j_client,
        generate_id, current_timestamp
    )


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class OrganizationConfig:
    """记忆组织配置"""
    # 组织策略
    use_temporal: bool = True       # 使用时间结构
    use_event: bool = True          # 使用事件结构
    use_semantic: bool = True       # 使用语义结构
    use_hierarchy: bool = True      # 使用层级抽象
    
    # 时间结构参数
    time_threshold_minutes: int = 30      # 会话切分的时间阈值
    topic_drift_threshold: float = 0.3    # 主题漂移阈值
    
    # 语义结构参数
    similarity_threshold: float = 0.4     # 相似度阈值
    max_candidates: int = 50              # 最大候选数
    
    # 层级抽象参数
    min_cluster_size: int = 2             # 最小聚类大小
    abstraction_threshold: float = 0.5    # 抽象阈值
    
    # 存储配置
    auto_save_edges: bool = True          # 是否自动保存边
    
    # 调试参数
    verbose: bool = False


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class Session:
    """会话/片段：时间连续的记忆集合"""
    session_id: str
    memory_ids: List[str]
    start_time: str
    end_time: str
    topic: Optional[str] = None


@dataclass
class Phase:
    """阶段：更高层次的时间划分"""
    phase_id: str
    label: str
    session_ids: List[str]
    start_time: str
    end_time: str
    summary: Optional[str] = None


@dataclass
class EventUnit:
    """事件单元：结构化的事件表示"""
    event_id: str
    agent: Optional[str]        # 执行主体
    action: str                 # 动作
    object: Optional[str]       # 作用对象
    outcome: Optional[str]      # 结果
    context: Optional[str]      # 上下文
    timestamp: str
    source_memory_id: str
    confidence: float = 0.9


@dataclass
class AbstractionNode:
    """抽象节点：从具体案例中提炼的模式"""
    node_id: str
    label: str
    condition: str
    solution: str
    verification: str
    support_ids: List[str]
    confidence: float = 0.8


@dataclass
class MemoryGraph:
    """记忆图谱：整合所有组织结构的完整图"""
    memories: Dict[str, MemoryItem]
    edges: List[Edge]
    sessions: List[Session]
    phases: List[Phase]
    abstractions: Dict[str, AbstractionNode]
    
    def get_node_count(self) -> int:
        return len(self.memories)
    
    def get_edge_count(self) -> int:
        return len(self.edges)
    
    def get_edges_by_type(self, relation_type: str) -> List[Edge]:
        return [e for e in self.edges if e.relation_type == relation_type]
    
    def get_neighbors(self, node_id: str) -> List[str]:
        neighbors = set()
        for edge in self.edges:
            if edge.source_id == node_id:
                neighbors.add(edge.target_id)
            elif edge.target_id == node_id:
                neighbors.add(edge.source_id)
        return list(neighbors)


# =============================================================================
# 时间结构构建器
# =============================================================================

class TemporalStructureBuilder:
    """
    时间结构构建器
    负责将记忆按时间维度组织成会话和阶段
    """
    
    def __init__(self, config: OrganizationConfig):
        self.config = config
        self.embedding = get_embedding_client()
        self.sessions: List[Session] = []
        self.phases: List[Phase] = []
        self.memory_to_session: Dict[str, str] = {}
    
    def build(self, memories: List[MemoryItem]) -> Tuple[List[Session], List[Phase], List[Edge]]:
        """
        构建时间结构
        
        Returns:
            (会话列表, 阶段列表, 时间边列表)
        """
        if not memories:
            return [], [], []
        
        # 1. 会话切分
        self.sessions = self._segment_into_sessions(memories)
        
        # 2. 阶段构建
        self.phases = self._build_phases(self.sessions)
        
        # 3. 生成时间边
        edges = self._get_temporal_edges(memories)
        
        if self.config.verbose:
            print(f"[TemporalStructureBuilder] 会话: {len(self.sessions)}, 阶段: {len(self.phases)}, 边: {len(edges)}")
        
        return self.sessions, self.phases, edges
    
    def _segment_into_sessions(self, memories: List[MemoryItem]) -> List[Session]:
        """会话切分：基于时间间隔和主题漂移"""
        # 按时间排序
        sorted_memories = sorted(memories, key=lambda m: m.created_at)
        
        sessions = []
        current_session_memories = [sorted_memories[0]]
        
        for i in range(1, len(sorted_memories)):
            prev_memory = sorted_memories[i - 1]
            curr_memory = sorted_memories[i]
            
            # 计算时间间隔
            try:
                prev_time = datetime.fromisoformat(prev_memory.created_at.replace('Z', ''))
                curr_time = datetime.fromisoformat(curr_memory.created_at.replace('Z', ''))
                time_gap = (curr_time - prev_time).total_seconds() / 60
            except:
                time_gap = 0
            
            # 判断是否切分
            time_break = time_gap > self.config.time_threshold_minutes
            
            # 主题漂移检测（使用tag重叠度）
            prev_tags = set(prev_memory.tags)
            curr_tags = set(curr_memory.tags)
            overlap = len(prev_tags & curr_tags) / max(len(prev_tags | curr_tags), 1)
            topic_break = overlap < self.config.topic_drift_threshold
            
            if time_break or topic_break:
                # 保存当前会话
                session = self._create_session(current_session_memories)
                sessions.append(session)
                current_session_memories = [curr_memory]
            else:
                current_session_memories.append(curr_memory)
        
        # 保存最后一个会话
        if current_session_memories:
            session = self._create_session(current_session_memories)
            sessions.append(session)
        
        # 建立映射
        for session in sessions:
            for mem_id in session.memory_ids:
                self.memory_to_session[mem_id] = session.session_id
        
        return sessions
    
    def _create_session(self, memories: List[MemoryItem]) -> Session:
        """创建会话对象"""
        memory_ids = [m.id for m in memories]
        start_time = min(m.created_at for m in memories)
        end_time = max(m.created_at for m in memories)
        
        # 提取主题
        all_tags = []
        for m in memories:
            all_tags.extend(m.tags)
        topic = max(set(all_tags), key=all_tags.count) if all_tags else None
        
        return Session(
            session_id=generate_id(),
            memory_ids=memory_ids,
            start_time=start_time,
            end_time=end_time,
            topic=topic
        )
    
    def _build_phases(self, sessions: List[Session]) -> List[Phase]:
        """阶段构建：将会话归并为更高层次的阶段"""
        if not sessions:
            return []
        
        # 简化：每天作为一个阶段
        day_groups = defaultdict(list)
        for session in sessions:
            try:
                day = datetime.fromisoformat(session.start_time.replace('Z', '')).date()
                day_groups[day].append(session)
            except:
                pass
        
        phases = []
        for day, day_sessions in sorted(day_groups.items()):
            phase = Phase(
                phase_id=generate_id(),
                label=f"Phase_{day.isoformat()}",
                session_ids=[s.session_id for s in day_sessions],
                start_time=min(s.start_time for s in day_sessions),
                end_time=max(s.end_time for s in day_sessions),
                summary=f"包含{len(day_sessions)}个会话"
            )
            phases.append(phase)
        
        return phases
    
    def _get_temporal_edges(self, memories: List[MemoryItem]) -> List[Edge]:
        """生成时间关系边"""
        edges = []
        sorted_memories = sorted(memories, key=lambda m: m.created_at)
        
        for i in range(len(sorted_memories) - 1):
            curr = sorted_memories[i]
            next_mem = sorted_memories[i + 1]
            
            # 检查时间间隔
            try:
                curr_time = datetime.fromisoformat(curr.created_at.replace('Z', ''))
                next_time = datetime.fromisoformat(next_mem.created_at.replace('Z', ''))
                time_gap_minutes = (next_time - curr_time).total_seconds() / 60
            except:
                time_gap_minutes = 0
            
            # 在时间阈值内建立边
            if time_gap_minutes <= self.config.time_threshold_minutes * 2:
                same_session = self.memory_to_session.get(curr.id) == self.memory_to_session.get(next_mem.id)
                weight = 0.9 if same_session else 0.6
                
                edge = Edge(
                    source_id=curr.id,
                    target_id=next_mem.id,
                    relation_type=RelationType.FOLLOWS.value,
                    weight=weight,
                    metadata={"type": "temporal", "same_session": same_session}
                )
                edges.append(edge)
        
        return edges


# =============================================================================
# 事件结构构建器
# =============================================================================

class EventStructureBuilder:
    """
    事件结构构建器
    负责从记忆中抽取事件要素并建立事件间关系
    """
    
    def __init__(self, config: OrganizationConfig):
        self.config = config
        self.llm = get_llm_client()
        self.events: Dict[str, EventUnit] = {}
        self.memory_to_event: Dict[str, str] = {}
    
    def build(self, memories: List[MemoryItem]) -> Tuple[List[EventUnit], List[Edge]]:
        """
        构建事件结构
        
        Returns:
            (事件列表, 事件关系边列表)
        """
        # 1. 抽取事件
        events = self._extract_events(memories)
        
        # 2. 推断事件关系
        edges = self._infer_event_relations(events)
        
        if self.config.verbose:
            print(f"[EventStructureBuilder] 事件: {len(events)}, 边: {len(edges)}")
        
        return events, edges
    
    def _extract_events(self, memories: List[MemoryItem]) -> List[EventUnit]:
        """从记忆中抽取事件要素"""
        events = []
        
        for memory in memories:
            event = self._extract_single_event(memory)
            if event:
                events.append(event)
                self.events[event.event_id] = event
                self.memory_to_event[memory.id] = event.event_id
        
        return events
    
    def _extract_single_event(self, memory: MemoryItem) -> Optional[EventUnit]:
        """使用LLM抽取单个事件"""
        prompt = f"""从以下记忆中抽取事件要素。

记忆内容：{memory.value}

输出JSON格式：
{{
    "agent": "执行主体（如：用户、系统、某人）",
    "action": "动作",
    "object": "作用对象",
    "outcome": "结果",
    "context": "上下文"
}}

如果无法抽取有效事件，返回：{{"valid": false}}

只输出JSON。"""
        
        response = self.llm.chat(
            system_prompt="你是一个事件抽取专家。",
            user_prompt=prompt
        )
        
        result = self.llm.parse_json(response)
        
        if not result or result.get("valid") == False:
            # 使用简单规则抽取
            return self._rule_based_extract(memory)
        
        return EventUnit(
            event_id=generate_id(),
            agent=result.get("agent"),
            action=result.get("action", "记录"),
            object=result.get("object"),
            outcome=result.get("outcome"),
            context=result.get("context"),
            timestamp=memory.created_at,
            source_memory_id=memory.id
        )
    
    def _rule_based_extract(self, memory: MemoryItem) -> EventUnit:
        """基于规则的简单事件抽取"""
        text = memory.value
        
        # 简单规则
        agent = "用户"
        action = "记录"
        
        keywords = {
            "喜欢": ("用户", "表达偏好"),
            "不喜欢": ("用户", "表达偏好"),
            "完成": ("用户", "完成任务"),
            "计划": ("用户", "制定计划"),
            "工作": ("用户", "工作相关"),
        }
        
        for keyword, (ag, act) in keywords.items():
            if keyword in text:
                agent, action = ag, act
                break
        
        return EventUnit(
            event_id=generate_id(),
            agent=agent,
            action=action,
            object=None,
            outcome=None,
            context=None,
            timestamp=memory.created_at,
            source_memory_id=memory.id
        )
    
    def _infer_event_relations(self, events: List[EventUnit]) -> List[Edge]:
        """推断事件之间的关系"""
        edges = []
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        for i, event_a in enumerate(sorted_events):
            for j in range(i + 1, min(i + 5, len(sorted_events))):
                event_b = sorted_events[j]
                
                # 推断关系
                relation_type, confidence = self._infer_relation(event_a, event_b)
                
                if confidence >= 0.5:
                    edge = Edge(
                        source_id=event_a.source_memory_id,
                        target_id=event_b.source_memory_id,
                        relation_type=relation_type,
                        weight=confidence,
                        metadata={"type": "event"}
                    )
                    edges.append(edge)
        
        return edges
    
    def _infer_relation(self, event_a: EventUnit, event_b: EventUnit) -> Tuple[str, float]:
        """推断两个事件之间的关系"""
        action_a = event_a.action.lower() if event_a.action else ""
        action_b = event_b.action.lower() if event_b.action else ""
        
        # 简单规则推断
        if "计划" in action_a and ("完成" in action_b or "执行" in action_b):
            return RelationType.CAUSES.value, 0.8
        
        if "问题" in action_a and "解决" in action_b:
            return RelationType.RESOLVES.value, 0.85
        
        return RelationType.FOLLOWS.value, 0.6


# =============================================================================
# 语义结构构建器
# =============================================================================

class SemanticStructureBuilder:
    """
    语义结构构建器
    负责对记忆进行语义结构化和关系建模
    """
    
    def __init__(self, config: OrganizationConfig):
        self.config = config
        self.embedding = get_embedding_client()
    
    def build(self, memories: List[MemoryItem]) -> List[Edge]:
        """
        构建语义结构
        
        Returns:
            语义关系边列表
        """
        edges = []
        
        # 为每个记忆生成embedding（如果没有）
        for mem in memories:
            if not mem.embedding:
                text = f"{mem.key} {mem.value}"
                mem.embedding = self.embedding.embed(text)
        
        # 计算两两相似度并建立边
        for i, mem_a in enumerate(memories):
            for j in range(i + 1, len(memories)):
                mem_b = memories[j]
                
                # 计算相似度
                similarity = self.embedding.similarity(mem_a.embedding, mem_b.embedding)
                
                if similarity >= self.config.similarity_threshold:
                    # 判断关系类型
                    relation_type = self._classify_relation(mem_a, mem_b, similarity)
                    
                    edge = Edge(
                        source_id=mem_a.id,
                        target_id=mem_b.id,
                        relation_type=relation_type,
                        weight=similarity,
                        metadata={"type": "semantic"}
                    )
                    edges.append(edge)
        
        if self.config.verbose:
            print(f"[SemanticStructureBuilder] 边: {len(edges)}")
        
        return edges
    
    def _classify_relation(self, mem_a: MemoryItem, mem_b: MemoryItem, 
                          similarity: float) -> str:
        """分类关系类型"""
        # 检查tag重叠
        tags_a = set(mem_a.tags)
        tags_b = set(mem_b.tags)
        tag_overlap = len(tags_a & tags_b) / max(len(tags_a | tags_b), 1)
        
        if similarity >= 0.8 and tag_overlap >= 0.5:
            return RelationType.SAME_TOPIC.value
        
        return RelationType.RELATED_TO.value


# =============================================================================
# 层级抽象构建器
# =============================================================================

class HierarchyBuilder:
    """
    层级抽象构建器
    负责构建索引树和生成抽象节点
    """
    
    def __init__(self, config: OrganizationConfig):
        self.config = config
        self.llm = get_llm_client()
        self.embedding = get_embedding_client()
        self.abstractions: Dict[str, AbstractionNode] = {}
    
    def build(self, memories: List[MemoryItem], 
              events: List[EventUnit] = None) -> Tuple[Dict[str, AbstractionNode], List[Edge]]:
        """
        构建层级抽象
        
        Returns:
            (抽象节点字典, 抽象关系边列表)
        """
        # 1. 聚合相似记忆
        clusters = self._aggregate_similar(memories)
        
        # 2. 生成抽象
        for cluster in clusters:
            if len(cluster) >= self.config.min_cluster_size:
                abstraction = self._generate_abstraction(cluster, events or [])
                if abstraction:
                    self.abstractions[abstraction.node_id] = abstraction
        
        # 3. 生成抽象关系边
        edges = self._get_abstraction_edges()
        
        if self.config.verbose:
            print(f"[HierarchyBuilder] 抽象: {len(self.abstractions)}, 边: {len(edges)}")
        
        return self.abstractions, edges
    
    def _aggregate_similar(self, memories: List[MemoryItem]) -> List[List[MemoryItem]]:
        """聚合相似记忆"""
        clusters = []
        used = set()
        
        for i, mem in enumerate(memories):
            if mem.id in used:
                continue
            
            cluster = [mem]
            used.add(mem.id)
            
            for j in range(i + 1, len(memories)):
                other = memories[j]
                if other.id in used:
                    continue
                
                # 计算相似度
                if mem.embedding and other.embedding:
                    sim = self.embedding.similarity(mem.embedding, other.embedding)
                    if sim >= self.config.abstraction_threshold:
                        cluster.append(other)
                        used.add(other.id)
            
            if len(cluster) >= self.config.min_cluster_size:
                clusters.append(cluster)
        
        return clusters
    
    def _generate_abstraction(self, memories: List[MemoryItem], 
                              events: List[EventUnit]) -> Optional[AbstractionNode]:
        """从记忆簇生成抽象模式"""
        # 收集记忆内容
        contents = [m.value for m in memories]
        
        prompt = f"""从以下相关记忆中提炼一个通用模式/规律。

记忆内容：
{chr(10).join(f"- {c}" for c in contents)}

输出JSON格式：
{{
    "label": "模式名称",
    "condition": "适用条件",
    "solution": "建议方案",
    "verification": "验证方法"
}}

只输出JSON。"""
        
        response = self.llm.chat(
            system_prompt="你是一个模式归纳专家。",
            user_prompt=prompt
        )
        
        result = self.llm.parse_json(response)
        
        if not result:
            return None
        
        return AbstractionNode(
            node_id=generate_id(),
            label=result.get("label", "未知模式"),
            condition=result.get("condition", ""),
            solution=result.get("solution", ""),
            verification=result.get("verification", ""),
            support_ids=[m.id for m in memories]
        )
    
    def _get_abstraction_edges(self) -> List[Edge]:
        """生成抽象关系边"""
        edges = []
        
        for abs_id, abstraction in self.abstractions.items():
            for support_id in abstraction.support_ids:
                edge = Edge(
                    source_id=abs_id,
                    target_id=support_id,
                    relation_type=RelationType.CONTAINS.value,
                    weight=abstraction.confidence,
                    metadata={"type": "abstraction", "pattern_label": abstraction.label}
                )
                edges.append(edge)
        
        return edges


# =============================================================================
# 记忆组织主类
# =============================================================================

class MemoryOrganizer:
    """
    记忆组织器：整合所有组织模块的主入口
    
    使用方式：
        organizer = MemoryOrganizer(config)
        graph = organizer.run(memories)
    """
    
    def __init__(self, config: OrganizationConfig = None):
        self.config = config or OrganizationConfig()
        
        # 子模块
        self.temporal_builder = TemporalStructureBuilder(self.config)
        self.event_builder = EventStructureBuilder(self.config)
        self.semantic_builder = SemanticStructureBuilder(self.config)
        self.hierarchy_builder = HierarchyBuilder(self.config)
        
        # 数据库客户端
        self.neo4j = get_neo4j_client()
        
        if self.config.verbose:
            print(f"[MemoryOrganizer] 初始化完成")
    
    def run(self, memories: List[MemoryItem]) -> MemoryGraph:
        """
        执行完整的记忆组织流程
        
        Args:
            memories: 记忆列表
            
        Returns:
            组织好的记忆图谱
        """
        all_edges = []
        sessions = []
        phases = []
        abstractions = {}
        events = []
        
        # 1. 时间结构
        if self.config.use_temporal:
            sessions, phases, temporal_edges = self.temporal_builder.build(memories)
            all_edges.extend(temporal_edges)
        
        # 2. 事件结构
        if self.config.use_event:
            events, event_edges = self.event_builder.build(memories)
            all_edges.extend(event_edges)
        
        # 3. 语义结构
        if self.config.use_semantic:
            semantic_edges = self.semantic_builder.build(memories)
            all_edges.extend(semantic_edges)
        
        # 4. 层级抽象
        if self.config.use_hierarchy:
            abstractions, abstraction_edges = self.hierarchy_builder.build(memories, events)
            all_edges.extend(abstraction_edges)
        
        # 5. 去重边
        unique_edges = self._deduplicate_edges(all_edges)
        
        # 6. 保存边到数据库
        if self.config.auto_save_edges:
            self._save_edges(unique_edges)
        
        # 构建图谱
        graph = MemoryGraph(
            memories={m.id: m for m in memories},
            edges=unique_edges,
            sessions=sessions,
            phases=phases,
            abstractions=abstractions
        )
        
        if self.config.verbose:
            print(f"[MemoryOrganizer] 完成: 节点={graph.get_node_count()}, 边={graph.get_edge_count()}")
        
        return graph
    
    def _deduplicate_edges(self, edges: List[Edge]) -> List[Edge]:
        """去重边（保留权重最高的）"""
        edge_map = {}
        for edge in edges:
            key = (edge.source_id, edge.target_id, edge.relation_type)
            if key not in edge_map or edge.weight > edge_map[key].weight:
                edge_map[key] = edge
        return list(edge_map.values())
    
    def _save_edges(self, edges: List[Edge]):
        """保存边到数据库"""
        for edge in edges:
            self.neo4j.save_edge(edge)


# =============================================================================
# 便捷函数
# =============================================================================

def organize_memories(
    memories: List[MemoryItem],
    use_temporal: bool = True,
    use_event: bool = True,
    use_semantic: bool = True,
    use_hierarchy: bool = True,
    verbose: bool = False
) -> MemoryGraph:
    """
    便捷函数：组织记忆
    
    Args:
        memories: 记忆列表
        use_temporal: 使用时间结构
        use_event: 使用事件结构
        use_semantic: 使用语义结构
        use_hierarchy: 使用层级抽象
        verbose: 是否打印调试信息
        
    Returns:
        记忆图谱
    """
    config = OrganizationConfig(
        use_temporal=use_temporal,
        use_event=use_event,
        use_semantic=use_semantic,
        use_hierarchy=use_hierarchy,
        verbose=verbose
    )
    organizer = MemoryOrganizer(config)
    return organizer.run(memories)


# =============================================================================
# 示例代码
# =============================================================================

def example_organization():
    """记忆组织示例"""
    print("=" * 60)
    print("记忆组织示例")
    print("=" * 60)
    
    # 创建测试记忆
    base_time = datetime(2025, 1, 7, 9, 0, 0)
    
    test_memories = [
        MemoryItem(
            id=generate_id(),
            key="网关延迟告警",
            value="api-gateway服务的p99延迟从50ms飙升到500ms，触发告警",
            memory_type="LongTermMemory",
            tags=["告警", "延迟", "api-gateway"],
            created_at=(base_time + timedelta(minutes=0)).isoformat()
        ),
        MemoryItem(
            id=generate_id(),
            key="初步排查日志",
            value="查看日志发现大量数据库连接超时错误",
            memory_type="LongTermMemory",
            tags=["排查", "日志", "数据库"],
            created_at=(base_time + timedelta(minutes=5)).isoformat()
        ),
        MemoryItem(
            id=generate_id(),
            key="连接池扩容",
            value="将数据库连接池从100扩容至200，并重启服务",
            memory_type="LongTermMemory",
            tags=["操作", "扩容", "连接池"],
            created_at=(base_time + timedelta(minutes=25)).isoformat()
        ),
        MemoryItem(
            id=generate_id(),
            key="延迟恢复",
            value="p99延迟恢复到正常水平45ms，告警自动解除",
            memory_type="LongTermMemory",
            tags=["恢复", "延迟", "告警解除"],
            created_at=(base_time + timedelta(minutes=35)).isoformat()
        ),
    ]
    
    # 执行组织
    config = OrganizationConfig(
        use_temporal=True,
        use_event=True,
        use_semantic=True,
        use_hierarchy=True,
        auto_save_edges=False,  # 示例中不保存
        verbose=True
    )
    
    organizer = MemoryOrganizer(config)
    graph = organizer.run(test_memories)
    
    print(f"\n记忆图谱统计:")
    print(f"  节点数: {graph.get_node_count()}")
    print(f"  边数: {graph.get_edge_count()}")
    print(f"  会话数: {len(graph.sessions)}")
    print(f"  阶段数: {len(graph.phases)}")
    print(f"  抽象数: {len(graph.abstractions)}")
    
    print(f"\n边类型分布:")
    for rel_type in RelationType:
        count = len(graph.get_edges_by_type(rel_type.value))
        if count > 0:
            print(f"  {rel_type.value}: {count}")
    
    return graph


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    example_organization()
