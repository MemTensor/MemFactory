# =============================================================================
# 记忆更新模块 - Memory Update Module
# 本代码用于书籍《Memory Engineering》第13章配套代码
# 涵盖：覆盖、合并、版本化、冲突消解、遗忘策略
# =============================================================================

"""
## Objective / 目标
- 实现记忆更新的核心操作：覆盖、合并、版本化
- 实现冲突消解机制
- 实现遗忘策略与长期稳定性维护

## I/O Spec / 输入输出约定
| Name | Type | Description |
|------|------|-------------|
| input | MemoryItem / List[MemoryItem] | 待更新的记忆 |
| config | UpdateConfig | 更新配置 |
| result | UpdateResult | 更新结果 |
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Literal
from datetime import datetime, timedelta
from enum import Enum
import math
import copy

try:
    from .common import (
        MemoryItem, MemoryStatus, UpdateAction,
        get_llm_client, get_embedding_client, get_neo4j_client, get_milvus_client,
        generate_id, current_timestamp
    )
except ImportError:
    from common import (
        MemoryItem, MemoryStatus, UpdateAction,
        get_llm_client, get_embedding_client, get_neo4j_client, get_milvus_client,
        generate_id, current_timestamp
    )


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class UpdateConfig:
    """记忆更新配置"""
    # 更新策略
    strategy: Literal["overwrite", "merge", "version", "auto"] = "auto"
    
    # 衰减参数
    decay_rate: float = 0.1              # 基础衰减率
    decay_interval_hours: int = 24       # 衰减计算间隔
    min_decay_score: float = 0.1         # 最小衰减分数
    
    # 遗忘阈值
    archive_threshold: float = 0.3       # 归档阈值
    delete_threshold: float = 0.1        # 删除阈值
    
    # 合并参数
    similarity_threshold: float = 0.7    # 相似度阈值
    merge_confidence_boost: float = 0.1  # 合并后置信度提升
    
    # 冲突消解参数
    time_priority_weight: float = 0.4    # 时效性权重
    source_priority_weight: float = 0.4  # 来源可信度权重
    
    # 存储配置
    auto_save: bool = True               # 是否自动保存
    
    # 调试参数
    verbose: bool = False


class ConflictType(Enum):
    """冲突类型"""
    FACTUAL = "factual"          # 事实冲突
    PREFERENCE = "preference"    # 偏好冲突
    TEMPORAL = "temporal"        # 时间冲突


@dataclass
class ConflictRecord:
    """冲突记录"""
    conflict_id: str
    memory_id_a: str
    memory_id_b: str
    conflict_type: ConflictType
    description: str
    resolution: Optional[str] = None
    resolved: bool = False
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = current_timestamp()


@dataclass
class UpdateResult:
    """更新结果"""
    action: str                          # 执行的操作
    success: bool                        # 是否成功
    memory_id: str                       # 结果记忆ID
    original_ids: List[str] = field(default_factory=list)  # 原始记忆ID
    conflicts: List[ConflictRecord] = field(default_factory=list)  # 冲突记录
    message: str = ""                    # 消息


# =============================================================================
# 记忆覆盖器
# =============================================================================

class MemoryOverwriter:
    """
    记忆覆盖器：用新信息完全替代旧信息
    
    适用场景：
    1. 用户明确修改偏好
    2. 配置项被替换
    3. 明确纠错
    """
    
    def __init__(self, config: UpdateConfig):
        self.config = config
    
    def overwrite(self, old_memory: MemoryItem, new_content: str,
                  new_key: str = None, new_tags: List[str] = None,
                  reason: str = "用户主动更新") -> MemoryItem:
        """
        覆盖记忆内容
        
        Args:
            old_memory: 原记忆
            new_content: 新内容
            new_key: 新关键词（可选）
            new_tags: 新标签（可选）
            reason: 覆盖原因
            
        Returns:
            更新后的记忆节点
        """
        now = current_timestamp()
        
        updated_memory = MemoryItem(
            id=old_memory.id,  # 保留原ID
            key=new_key or old_memory.key,
            value=new_content,
            memory_type=old_memory.memory_type,
            tags=new_tags or old_memory.tags,
            confidence=old_memory.confidence,
            created_at=old_memory.created_at,
            updated_at=now,
            user_id=old_memory.user_id,
            session_id=old_memory.session_id,
            status=MemoryStatus.ACTIVATED.value,
            source_type=old_memory.source_type,
            source_credibility=old_memory.source_credibility,
            access_count=old_memory.access_count,
            decay_score=1.0,  # 重置衰减分数
            version=old_memory.version + 1,
            embedding=None  # 需要重新生成
        )
        
        if self.config.verbose:
            print(f"[MemoryOverwriter] 覆盖记忆 {old_memory.id}")
            print(f"  原内容: {old_memory.value[:50]}...")
            print(f"  新内容: {new_content[:50]}...")
        
        return updated_memory
    
    def deprecate(self, memory: MemoryItem, reason: str = "信息过时") -> MemoryItem:
        """将记忆标记为废弃"""
        deprecated_memory = copy.deepcopy(memory)
        deprecated_memory.status = MemoryStatus.DEPRECATED.value
        deprecated_memory.updated_at = current_timestamp()
        
        if self.config.verbose:
            print(f"[MemoryOverwriter] 废弃记忆 {memory.id}")
        
        return deprecated_memory


# =============================================================================
# 记忆合并器
# =============================================================================

class MemoryMerger:
    """
    记忆合并器：将多条相似或互补的记忆合并为一条
    
    适用场景：
    1. 用户多次表达同一主题的偏好
    2. 同一事实的多个片段需要整合
    3. 重复记忆的去重与压缩
    """
    
    def __init__(self, config: UpdateConfig):
        self.config = config
        self.llm = get_llm_client()
        self.embedding = get_embedding_client()
    
    def should_merge(self, memory_a: MemoryItem, memory_b: MemoryItem) -> Tuple[bool, float]:
        """
        判断两条记忆是否应该合并
        
        Returns:
            (是否应该合并, 相似度)
        """
        # 必须是同一用户
        if memory_a.user_id != memory_b.user_id:
            return False, 0.0
        
        # 必须是同一类型
        if memory_a.memory_type != memory_b.memory_type:
            return False, 0.0
        
        # 计算相似度
        if memory_a.embedding and memory_b.embedding:
            similarity = self.embedding.similarity(memory_a.embedding, memory_b.embedding)
        else:
            emb_a = self.embedding.embed(f"{memory_a.key} {memory_a.value}")
            emb_b = self.embedding.embed(f"{memory_b.key} {memory_b.value}")
            similarity = self.embedding.similarity(emb_a, emb_b)
        
        should = similarity >= self.config.similarity_threshold
        return should, similarity
    
    def merge_two(self, memory_a: MemoryItem, memory_b: MemoryItem) -> MemoryItem:
        """合并两条记忆"""
        return self.merge_batch([memory_a, memory_b])
    
    def merge_batch(self, memories: List[MemoryItem]) -> MemoryItem:
        """
        批量合并多条记忆
        
        Args:
            memories: 待合并的记忆列表
            
        Returns:
            合并后的记忆
        """
        if not memories:
            raise ValueError("记忆列表不能为空")
        
        if len(memories) == 1:
            return memories[0]
        
        now = current_timestamp()
        
        # 使用LLM合并内容
        merged_content = self._llm_merge(memories)
        
        # 合并标签
        all_tags = []
        for m in memories:
            all_tags.extend(m.tags)
        merged_tags = list(set(all_tags))
        
        # 选择最具代表性的key
        merged_key = max(memories, key=lambda m: len(m.key)).key
        
        # 置信度
        merged_confidence = min(
            max(m.confidence for m in memories) + self.config.merge_confidence_boost,
            1.0
        )
        
        # 可信度
        merged_credibility = max(m.source_credibility for m in memories)
        
        # 访问次数累加
        total_access = sum(m.access_count for m in memories)
        
        merged_memory = MemoryItem(
            id=generate_id(),
            key=merged_key,
            value=merged_content,
            memory_type=memories[0].memory_type,
            tags=merged_tags,
            confidence=merged_confidence,
            created_at=min(m.created_at for m in memories),
            updated_at=now,
            user_id=memories[0].user_id,
            session_id=memories[0].session_id,
            status=MemoryStatus.ACTIVATED.value,
            source_type="merged",
            source_credibility=merged_credibility,
            access_count=total_access,
            decay_score=1.0,
            version=1
        )
        
        if self.config.verbose:
            print(f"[MemoryMerger] 合并了 {len(memories)} 条记忆")
            print(f"  新记忆ID: {merged_memory.id}")
        
        return merged_memory
    
    def _llm_merge(self, memories: List[MemoryItem]) -> str:
        """使用LLM合并记忆内容"""
        prompt = f"""Merge the following {len(memories)} related memories into one concise and complete comprehensive memory.

Memory content:
{chr(10).join(f"- [{m.created_at[:10]}] {m.value}" for m in sorted(memories, key=lambda x: x.created_at))}

Principles:
1. Retain all important information, remove redundancy
2. Organize chronologically
3. Use latest information as reference
4. Maintain objective conciseness

Output the merged memory content directly:"""
        
        response = self.llm.chat(
            system_prompt="You are a memory integration expert.",
            user_prompt=prompt
        )
        
        if not response:
            # 回退：简单拼接
            contents = [m.value for m in sorted(memories, key=lambda x: x.created_at)]
            return "Comprehensive memory: " + "；".join(contents)
        
        return response.strip()


# =============================================================================
# 版本管理器
# =============================================================================

class MemoryVersionManager:
    """
    记忆版本管理器：在不删除历史的情况下记录记忆的变化轨迹
    
    适用场景：
    1. 企业文档知识库的版本迭代
    2. 需要审计追溯的场景
    3. 误更新后需要回滚的场景
    """
    
    def __init__(self, config: UpdateConfig):
        self.config = config
        # 版本链存储：root_id -> [version_ids]
        self.version_chains: Dict[str, List[str]] = {}
    
    def create_version(self, old_memory: MemoryItem, new_content: str,
                       change_description: str) -> Tuple[MemoryItem, MemoryItem]:
        """
        创建新版本
        
        Args:
            old_memory: 原记忆
            new_content: 新内容
            change_description: 变更描述
            
        Returns:
            (归档的旧版本, 新版本)
        """
        now = current_timestamp()
        
        # 归档旧版本
        archived_old = copy.deepcopy(old_memory)
        archived_old.status = MemoryStatus.ARCHIVED.value
        archived_old.updated_at = now
        
        # 创建新版本
        new_version = MemoryItem(
            id=generate_id(),
            key=old_memory.key,
            value=new_content,
            memory_type=old_memory.memory_type,
            tags=old_memory.tags,
            confidence=old_memory.confidence,
            created_at=now,
            updated_at=now,
            user_id=old_memory.user_id,
            session_id=old_memory.session_id,
            status=MemoryStatus.ACTIVATED.value,
            source_type=old_memory.source_type,
            source_credibility=old_memory.source_credibility,
            access_count=0,
            decay_score=1.0,
            version=old_memory.version + 1
        )
        
        # 更新版本链
        root_id = self._find_root_id(old_memory.id)
        if root_id not in self.version_chains:
            self.version_chains[root_id] = [old_memory.id]
        self.version_chains[root_id].append(new_version.id)
        
        if self.config.verbose:
            print(f"[MemoryVersionManager] 创建新版本")
            print(f"  旧版本: {old_memory.id} (v{old_memory.version}) -> 已归档")
            print(f"  新版本: {new_version.id} (v{new_version.version})")
        
        return archived_old, new_version
    
    def _find_root_id(self, memory_id: str) -> str:
        """查找版本链的根ID"""
        for root_id, chain in self.version_chains.items():
            if memory_id in chain:
                return root_id
        return memory_id
    
    def get_version_history(self, memory_id: str) -> List[str]:
        """获取版本历史"""
        root_id = self._find_root_id(memory_id)
        return self.version_chains.get(root_id, [memory_id])


# =============================================================================
# 冲突消解器
# =============================================================================

class ConflictResolver:
    """
    冲突消解器：检测并解决记忆之间的冲突
    
    冲突类型：
    1. 显式事实冲突：硬性矛盾
    2. 隐式偏好冲突：行为偏离
    """
    
    def __init__(self, config: UpdateConfig):
        self.config = config
        self.llm = get_llm_client()
        self.conflict_records: List[ConflictRecord] = []
    
    def detect_conflict(self, new_memory: MemoryItem,
                        existing_memories: List[MemoryItem]) -> List[ConflictRecord]:
        """检测冲突"""
        conflicts = []
        
        for existing in existing_memories:
            if existing.status != MemoryStatus.ACTIVATED.value:
                continue
            
            conflict_info = self._check_conflict(new_memory, existing)
            
            if conflict_info:
                record = ConflictRecord(
                    conflict_id=generate_id(),
                    memory_id_a=new_memory.id,
                    memory_id_b=existing.id,
                    conflict_type=conflict_info["type"],
                    description=conflict_info["description"]
                )
                conflicts.append(record)
                self.conflict_records.append(record)
        
        return conflicts
    
    def _check_conflict(self, mem_a: MemoryItem, mem_b: MemoryItem) -> Optional[Dict]:
        """使用LLM检测冲突"""
        prompt = f"""Determine whether the following two memories have logical conflicts.

Memory A: {mem_a.value} (Time: {mem_a.created_at})
Memory B: {mem_b.value} (Time: {mem_b.created_at})

Output JSON format:
- With conflict: {{"has_conflict": true, "conflict_type": "FACTUAL/PREFERENCE/TEMPORAL", "description": "Conflict description"}}
- No conflict: {{"has_conflict": false}}

Output JSON only."""
        
        response = self.llm.chat(
            system_prompt="You are a memory conflict detection expert.",
            user_prompt=prompt
        )
        
        result = self.llm.parse_json(response)
        
        if result and result.get("has_conflict"):
            type_map = {
                "FACTUAL": ConflictType.FACTUAL,
                "PREFERENCE": ConflictType.PREFERENCE,
                "TEMPORAL": ConflictType.TEMPORAL
            }
            return {
                "type": type_map.get(result.get("conflict_type", "FACTUAL"), ConflictType.FACTUAL),
                "description": result.get("description", "Conflict detected")
            }
        
        return None
    
    def resolve(self, conflict: ConflictRecord,
                memory_store: Dict[str, MemoryItem]) -> Dict:
        """
        解决冲突
        
        Returns:
            解决方案
        """
        mem_a = memory_store.get(conflict.memory_id_a)
        mem_b = memory_store.get(conflict.memory_id_b)
        
        if not mem_a or not mem_b:
            return {"error": "Memory does not exist"}
        
        # 使用LLM解决冲突
        prompt = f"""Resolve the following memory conflict.

Memory A: {mem_a.value}
- Updated time: {mem_a.updated_at}
- Source credibility: {mem_a.source_credibility}

Memory B: {mem_b.value}
- Updated time: {mem_b.updated_at}
- Source credibility: {mem_b.source_credibility}

Rules:
1. User explicit statement > System inference
2. When source credibility is similar, use the latest memory
3. Suggest merging when complementary

Output JSON format:
{{"action": "keep_a/keep_b/merge/coexist", "reason": "Reason"}}

Output JSON only."""
        
        response = self.llm.chat(
            system_prompt="You are a memory conflict arbitration expert.",
            user_prompt=prompt
        )
        
        result = self.llm.parse_json(response)
        
        if result:
            action = result.get("action", "keep_b")
            reason = result.get("reason", "LLM decision")
            
            if action == "keep_a":
                return {"action": "keep", "keep_id": mem_a.id, "deprecate_id": mem_b.id, "reason": reason}
            elif action == "keep_b":
                return {"action": "keep", "keep_id": mem_b.id, "deprecate_id": mem_a.id, "reason": reason}
            elif action == "merge":
                return {"action": "merge", "memory_ids": [mem_a.id, mem_b.id], "reason": reason}
            else:
                return {"action": "coexist", "reason": reason}
        
        # 默认：时效性优先
        if mem_a.updated_at > mem_b.updated_at:
            return {"action": "keep", "keep_id": mem_a.id, "deprecate_id": mem_b.id, "reason": "Timeliness priority"}
        return {"action": "keep", "keep_id": mem_b.id, "deprecate_id": mem_a.id, "reason": "Timeliness priority"}


# =============================================================================
# 遗忘策略器
# =============================================================================

class MemoryForgetter:
    """
    记忆遗忘器：实现工程化的遗忘机制
    
    遗忘动作（风险从低到高）：
    1. 降权：降低检索优先级
    2. 归档：从在线索引迁移到冷存储
    3. 删除：彻底移除
    """
    
    def __init__(self, config: UpdateConfig):
        self.config = config
    
    def calculate_decay(self, memory: MemoryItem,
                        current_time: datetime = None) -> float:
        """
        计算记忆的衰减分数
        
        衰减公式（类似艾宾浩斯曲线）：
        decay_score = e^(-decay_rate * hours / 24) * (1 + log(1 + access_count))
        """
        if current_time is None:
            current_time = datetime.now()
        
        # 计算时间差
        try:
            last_time = datetime.fromisoformat(memory.updated_at.replace('Z', ''))
            hours_elapsed = (current_time - last_time).total_seconds() / 3600
        except:
            hours_elapsed = 0
        
        # 时间衰减
        time_decay = math.exp(-self.config.decay_rate * hours_elapsed / 24)
        
        # 访问次数加成
        access_boost = 1 + math.log(1 + memory.access_count)
        
        # 最终分数
        decay_score = min(time_decay * access_boost, 1.0)
        decay_score = max(decay_score, self.config.min_decay_score)
        
        return decay_score
    
    def update_decay_scores(self, memories: List[MemoryItem]) -> List[MemoryItem]:
        """批量更新衰减分数"""
        current_time = datetime.now()
        
        for memory in memories:
            if memory.status == MemoryStatus.ACTIVATED.value:
                memory.decay_score = self.calculate_decay(memory, current_time)
        
        return memories
    
    def auto_cleanup(self, memories: List[MemoryItem]) -> Dict[str, List[MemoryItem]]:
        """
        自动清理：根据衰减分数执行分层遗忘
        
        Returns:
            {action: [memories]}
        """
        self.update_decay_scores(memories)
        
        result = {
            "kept": [],
            "archived": [],
            "deleted": []
        }
        
        for memory in memories:
            if memory.status != MemoryStatus.ACTIVATED.value:
                continue
            
            score = memory.decay_score
            
            if score >= self.config.archive_threshold:
                result["kept"].append(memory)
            elif score >= self.config.delete_threshold:
                memory.status = MemoryStatus.ARCHIVED.value
                result["archived"].append(memory)
            else:
                memory.status = MemoryStatus.ARCHIVED.value  # 保守策略
                result["archived"].append(memory)
        
        if self.config.verbose:
            print(f"[MemoryForgetter] 清理完成: 保留={len(result['kept'])}, 归档={len(result['archived'])}")
        
        return result
    
    def reinforce(self, memory: MemoryItem) -> MemoryItem:
        """强化：当记忆被访问时，增强其保留概率"""
        memory.access_count += 1
        memory.decay_score = min(memory.decay_score * 1.2, 1.0)
        
        if self.config.verbose:
            print(f"[MemoryForgetter] 强化记忆 {memory.id}, 访问次数={memory.access_count}")
        
        return memory


# =============================================================================
# 记忆更新主类
# =============================================================================

class MemoryUpdater:
    """
    记忆更新器：整合所有更新模块的主入口
    
    使用方式：
        updater = MemoryUpdater(config)
        result = updater.run(memory, action="merge", related_memories=[...])
    """
    
    def __init__(self, config: UpdateConfig = None):
        self.config = config or UpdateConfig()
        
        # 子模块
        self.overwriter = MemoryOverwriter(self.config)
        self.merger = MemoryMerger(self.config)
        self.version_manager = MemoryVersionManager(self.config)
        self.conflict_resolver = ConflictResolver(self.config)
        self.forgetter = MemoryForgetter(self.config)
        
        # 数据库客户端
        self.neo4j = get_neo4j_client()
        self.milvus = get_milvus_client()
        self.embedding = get_embedding_client()
        
        if self.config.verbose:
            print(f"[MemoryUpdater] 初始化完成，策略: {self.config.strategy}")
    
    def run(self, memory: MemoryItem,
            action: str = None,
            new_content: str = None,
            related_memories: List[MemoryItem] = None) -> UpdateResult:
        """
        执行记忆更新
        
        Args:
            memory: 待更新的记忆
            action: 操作类型 (overwrite/merge/version/auto)
            new_content: 新内容（用于覆盖/版本化）
            related_memories: 相关记忆（用于合并/冲突检测）
            
        Returns:
            更新结果
        """
        action = action or self.config.strategy
        
        # 自动选择策略
        if action == "auto":
            action = self._auto_select_action(memory, related_memories or [])
        
        if action == "overwrite" and new_content:
            return self._do_overwrite(memory, new_content)
        elif action == "merge" and related_memories:
            return self._do_merge([memory] + related_memories)
        elif action == "version" and new_content:
            return self._do_version(memory, new_content)
        else:
            return UpdateResult(
                action="none",
                success=False,
                memory_id=memory.id,
                message="无效的操作或参数"
            )
    
    def _auto_select_action(self, memory: MemoryItem,
                            related_memories: List[MemoryItem]) -> str:
        """自动选择更新策略"""
        if not related_memories:
            return "overwrite"
        
        # 检查是否有可合并的记忆
        for related in related_memories:
            should_merge, similarity = self.merger.should_merge(memory, related)
            if should_merge:
                return "merge"
        
        # 检查是否有冲突
        conflicts = self.conflict_resolver.detect_conflict(memory, related_memories)
        if conflicts:
            return "version"  # 有冲突时使用版本化
        
        return "overwrite"
    
    def decide_action(self, new_memory: MemoryItem,
                      related_memories: List[MemoryItem] = None) -> Dict[str, Any]:
        """
        决定对新记忆的操作（用于Pipeline调用）
        
        这是Pipeline流程中的核心决策函数：
        根据新抽取的记忆和已有相关记忆，决定应该执行什么操作
        
        Args:
            new_memory: 新抽取的候选记忆
            related_memories: 查询到的相关已有记忆
            
        Returns:
            决策结果字典:
            {
                "action": "ADD" | "MERGE" | "OVERWRITE" | "VERSION" | "IGNORE",
                "final_memory": MemoryItem,  # 最终要保存的记忆
                "deprecated_ids": List[str],  # 需要废弃的旧记忆ID
                "reason": str  # 决策原因
            }
        """
        related_memories = related_memories or []
        
        result = {
            "action": "ADD",
            "final_memory": new_memory,
            "deprecated_ids": [],
            "reason": ""
        }
        
        # 没有相关记忆，直接新增
        if not related_memories:
            result["action"] = "ADD"
            result["reason"] = "无相关已有记忆，直接新增"
            if self.config.verbose:
                print(f"[MemoryUpdater] 决策: ADD - {result['reason']}")
            return result
        
        # 检查是否有高度相似的记忆（可能是重复）
        highest_similarity = 0.0
        most_similar_memory = None
        
        for related in related_memories:
            should_merge, similarity = self.merger.should_merge(new_memory, related)
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_memory = related
        
        # 高度相似（>0.85）：可能是重复，忽略或合并
        if highest_similarity > 0.85:
            # 检查内容是否完全相同
            if new_memory.value.strip() == most_similar_memory.value.strip():
                result["action"] = "IGNORE"
                result["reason"] = f"与已有记忆完全相同 (相似度: {highest_similarity:.2f})"
            else:
                # 内容有差异，执行合并
                merged = self.merger.merge_two(most_similar_memory, new_memory)
                result["action"] = "MERGE"
                result["final_memory"] = merged
                result["deprecated_ids"] = [most_similar_memory.id]
                result["reason"] = f"与已有记忆高度相似 (相似度: {highest_similarity:.2f})，执行合并"
        
        # 中等相似（0.6-0.85）：检查是否有冲突
        elif highest_similarity > 0.6:
            conflicts = self.conflict_resolver.detect_conflict(new_memory, [most_similar_memory])
            
            if conflicts:
                # 有冲突，使用版本化或覆盖
                conflict_type = conflicts[0].conflict_type
                
                if conflict_type == ConflictType.FACTUAL:
                    # 事实冲突：保留新信息（用户最新输入）
                    result["action"] = "OVERWRITE"
                    result["final_memory"] = new_memory
                    result["deprecated_ids"] = [most_similar_memory.id]
                    result["reason"] = f"检测到事实冲突，以新信息覆盖"
                else:
                    # 偏好/时间冲突：版本化
                    _, new_version = self.version_manager.create_version(
                        most_similar_memory, new_memory.value, "信息更新"
                    )
                    result["action"] = "VERSION"
                    result["final_memory"] = new_version
                    result["deprecated_ids"] = [most_similar_memory.id]
                    result["reason"] = f"检测到{conflict_type.value}冲突，创建新版本"
            else:
                # 无冲突，合并补充信息
                merged = self.merger.merge_two(most_similar_memory, new_memory)
                result["action"] = "MERGE"
                result["final_memory"] = merged
                result["deprecated_ids"] = [most_similar_memory.id]
                result["reason"] = f"信息互补 (相似度: {highest_similarity:.2f})，执行合并"
        
        # 低相似（<0.6）：作为新记忆添加
        else:
            result["action"] = "ADD"
            result["reason"] = f"与已有记忆相似度较低 ({highest_similarity:.2f})，作为新记忆添加"
        
        if self.config.verbose:
            print(f"[MemoryUpdater] 决策: {result['action']} - {result['reason']}")
        
        return result
    
    def _do_overwrite(self, memory: MemoryItem, new_content: str) -> UpdateResult:
        """执行覆盖"""
        updated = self.overwriter.overwrite(memory, new_content)
        
        if self.config.auto_save:
            self._save_memory(updated)
        
        return UpdateResult(
            action="overwrite",
            success=True,
            memory_id=updated.id,
            original_ids=[memory.id],
            message="覆盖成功"
        )
    
    def _do_merge(self, memories: List[MemoryItem]) -> UpdateResult:
        """执行合并"""
        merged = self.merger.merge_batch(memories)
        
        if self.config.auto_save:
            # 废弃原记忆
            for mem in memories:
                deprecated = self.overwriter.deprecate(mem, "已合并")
                self.neo4j.save_memory(deprecated)
            
            # 保存新记忆
            self._save_memory(merged)
        
        return UpdateResult(
            action="merge",
            success=True,
            memory_id=merged.id,
            original_ids=[m.id for m in memories],
            message=f"合并了 {len(memories)} 条记忆"
        )
    
    def _do_version(self, memory: MemoryItem, new_content: str) -> UpdateResult:
        """执行版本化"""
        archived, new_version = self.version_manager.create_version(
            memory, new_content, "内容更新"
        )
        
        if self.config.auto_save:
            self.neo4j.save_memory(archived)
            self._save_memory(new_version)
        
        return UpdateResult(
            action="version",
            success=True,
            memory_id=new_version.id,
            original_ids=[memory.id],
            message=f"创建新版本 v{new_version.version}"
        )
    
    def _save_memory(self, memory: MemoryItem):
        """保存记忆到数据库"""
        # 生成embedding
        text = f"{memory.key} {memory.value}"
        memory.embedding = self.embedding.embed(text)
        
        # 保存到Milvus
        self.milvus.insert(memory.id, memory.embedding)
        
        # 保存到Neo4j
        self.neo4j.save_memory(memory)
    
    def cleanup(self, memories: List[MemoryItem]) -> Dict[str, List[MemoryItem]]:
        """执行清理任务"""
        result = self.forgetter.auto_cleanup(memories)
        
        if self.config.auto_save:
            for mem in result["archived"]:
                self.neo4j.save_memory(mem)
        
        return result
    
    def reinforce(self, memory: MemoryItem) -> MemoryItem:
        """强化记忆"""
        reinforced = self.forgetter.reinforce(memory)
        
        if self.config.auto_save:
            self.neo4j.save_memory(reinforced)
        
        return reinforced


# =============================================================================
# 便捷函数
# =============================================================================

def update_memory(
    memory: MemoryItem,
    action: str = "auto",
    new_content: str = None,
    related_memories: List[MemoryItem] = None,
    verbose: bool = False
) -> UpdateResult:
    """
    便捷函数：更新记忆
    
    Args:
        memory: 待更新的记忆
        action: 操作类型
        new_content: 新内容
        related_memories: 相关记忆
        verbose: 是否打印调试信息
        
    Returns:
        更新结果
    """
    config = UpdateConfig(
        strategy=action,
        verbose=verbose
    )
    updater = MemoryUpdater(config)
    return updater.run(memory, action, new_content, related_memories)


# =============================================================================
# 示例代码
# =============================================================================

def example_update():
    """记忆更新示例"""
    print("=" * 60)
    print("记忆更新示例")
    print("=" * 60)
    
    config = UpdateConfig(
        verbose=True,
        auto_save=False  # 示例中不保存
    )
    updater = MemoryUpdater(config)
    
    # 示例1：覆盖
    print("\n--- 示例1：覆盖 ---")
    original = MemoryItem(
        id=generate_id(),
        key="用户生日",
        value="用户的生日是3月2日",
        memory_type="UserMemory",
        tags=["个人信息"]
    )
    
    result = updater.run(
        original,
        action="overwrite",
        new_content="用户的生日是3月12日"
    )
    print(f"结果: {result.action}, 成功: {result.success}")
    
    # 示例2：合并
    print("\n--- 示例2：合并 ---")
    mem1 = MemoryItem(
        id=generate_id(),
        key="水果偏好",
        value="用户喜欢苹果",
        memory_type="UserMemory",
        tags=["偏好", "水果"]
    )
    mem2 = MemoryItem(
        id=generate_id(),
        key="水果偏好",
        value="用户喜欢香蕉",
        memory_type="UserMemory",
        tags=["偏好", "水果"]
    )
    
    result = updater.run(
        mem1,
        action="merge",
        related_memories=[mem2]
    )
    print(f"结果: {result.action}, 成功: {result.success}, 消息: {result.message}")
    
    # 示例3：版本化
    print("\n--- 示例3：版本化 ---")
    policy = MemoryItem(
        id=generate_id(),
        key="报销流程",
        value="报销需要纸质签字",
        memory_type="LongTermMemory",
        tags=["政策"]
    )
    
    result = updater.run(
        policy,
        action="version",
        new_content="报销已升级为全线上电子签名"
    )
    print(f"结果: {result.action}, 成功: {result.success}, 消息: {result.message}")
    
    return result


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    example_update()
