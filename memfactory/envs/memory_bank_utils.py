# =============================================================================
# Common configuration module
# Includes LLM, embedding, Neo4j, and Milvus clients.
# =============================================================================

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
from openai import OpenAI
# OpenAI API dependency
from ..common.utils import LLMClient


# =============================================================================
# Environment configuration
# Create a .env file at the project root if local overrides are needed.
# =============================================================================

# Try loading a .env file if one exists.
try:
    from dotenv import load_dotenv
    # Try loading .env from common project locations.
    for env_path in ['.env', '../.env', '../../.env']:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            break
except ImportError:
    pass  # python-dotenv is optional.

# OpenAI LLM API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-nano")

# Embedding API configuration with a separate endpoint and key.
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "EMPTY")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))  # BGE-M3 defaults to 1024 dimensions.

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")  # Leave empty to use the server default database.

# Milvus configuration
MILVUS_URI = os.getenv("MILVUS_URI", "")
MILVUS_USER = os.getenv("MILVUS_USER", "root")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "memory_embeddings")


# =============================================================================
# Enum definitions
# =============================================================================

class MemoryType(Enum):
    """Memory type enum."""
    LONG_TERM_MEMORY = "LongTermMemory"
    USER_MEMORY = "UserMemory"
    FACT = "fact"
    EVENT = "event"
    PREFERENCE = "preference"


class MemoryStatus(Enum):
    """Memory status enum."""
    ACTIVATED = "activated"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"
    DELETED = "deleted"


class UpdateAction(Enum):
    """Update action type."""
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    VERSION = "version"


class RelationType(Enum):
    """Relationship type enum."""
    CAUSES = "causes"
    FOLLOWS = "follows"
    RESOLVES = "resolves"
    CONTAINS = "contains"
    RELATED_TO = "related_to"
    SAME_TOPIC = "same_topic"
    DEPENDS_ON = "depends_on"


# =============================================================================
# Core data structures
# =============================================================================

@dataclass
class MemoryItem:
    """
    Memory item: the basic storage unit used across modules.
    """
    id: str
    key: str                          # Memory title or keyword.
    value: str                        # Memory content.
    memory_type: str                  # Memory type.
    tags: List[str]                   # Tag list.
    confidence: float = 0.9           # Confidence score in [0, 1].
    created_at: str = ""              # Creation timestamp.
    updated_at: str = ""              # Last update timestamp.
    user_id: str = "default_user"     # User ID.
    session_id: str = "default_session"  # Session ID.
    status: str = "activated"         # Status.
    source_type: str = "user_explicit"   # Source type.
    source_credibility: float = 1.0   # Source credibility.
    access_count: int = 0             # Access count.
    decay_score: float = 1.0          # Decay score.
    version: int = 1                  # Version number.
    embedding: Optional[List[float]] = None  # Vector representation.
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
    
    def to_dict(self) -> Dict:
        """Convert to a dictionary."""
        result = asdict(self)
        # Remove the raw embedding to reduce serialized output size.
        if 'embedding' in result and result['embedding'] is not None:
            result['embedding'] = f"<vector dim={len(result['embedding'])}>"
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryItem':
        """Create from a dictionary."""
        # Ignore serialized embedding placeholders.
        if 'embedding' in data and isinstance(data['embedding'], str):
            data['embedding'] = None
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ConversationMessage:
    """Conversation message."""
    role: str           # user / assistant / system
    content: str        # Message content.
    timestamp: Optional[str] = None # Timestamp.
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ExtractionResult:
    """Extraction result."""
    memory_list: List[MemoryItem]
    summary: str
    status: str = "SUCCESS"  # SUCCESS / BUFFERED / IGNORED / TIMEOUT


@dataclass
class SearchResult:
    """Search result."""
    memories: List[Tuple[MemoryItem, float]]  # (memory, relevance score)
    query: str
    total_found: int


@dataclass
class Edge:
    """Graph edge connecting two nodes."""
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# Embedding service
# =============================================================================

class EmbeddingClient:
    """
    Embedding client that calls an OpenAI-compatible embedding service.
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
        # Use the separate embedding service configuration.
        self.client = OpenAI(
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_BASE_URL
        )
        self.model = EMBEDDING_MODEL
        self.dim = EMBEDDING_DIM  # BGE-M3 defaults to 1024 dimensions.
        self._use_mock = True  # Use mock embeddings unless the API is enabled.
        self._initialized = True
        print(f"[EmbeddingClient] Initialized with model: {self.model}, endpoint: {EMBEDDING_BASE_URL}")
    
    def embed(self, text: str) -> List[float]:
        """
        Generate a vector representation for text.
        
        Args:
            text: Input text.
            
        Returns:
            Embedding vector.
        """
        if not self._use_mock:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"[EmbeddingClient] API call failed; using mock embeddings: {e}")
                self._use_mock = True
        
        # Mock implementation: deterministic vectors based on a text hash.
        return self._mock_embed(text)
    
    def _mock_embed(self, text: str) -> List[float]:
        """Mock embedding implementation."""
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        embedding = np.random.randn(self.dim).tolist()
        norm = np.linalg.norm(embedding)
        return [x / norm for x in embedding]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in a batch."""
        return [self.embed(text) for text in texts]
    
    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Compute cosine similarity."""
        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = np.sqrt(sum(a * a for a in emb1))
        norm2 = np.sqrt(sum(b * b for b in emb2))
        return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0


# =============================================================================
# Neo4j graph database client
# =============================================================================

class Neo4jClient:
    """
    Neo4j client for storing and querying the memory graph.
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
        self._driver = None
        self._use_mock = True
        self._mock_store: Dict[str, MemoryItem] = {}
        self._mock_edges: List[Edge] = []
        self._database = NEO4J_DATABASE
        
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            self._driver.verify_connectivity()
            
            # Ensure the target database exists when possible.
            self._ensure_database_exists()
            
            self._use_mock = False
            print(f"[Neo4jClient] Connected to {NEO4J_URI}, database: {self._database}")
        except Exception as e:
            print(f"[Neo4jClient] Connection failed; using in-memory storage: {e}")
        
        self._initialized = True
    
    def _ensure_database_exists(self):
        """Ensure the database exists, creating it when supported."""
        if not self._database:
            return
        
        try:
            # Use the system database to create a new database.
            with self._driver.session(database="system") as session:
                # Check whether the database already exists.
                result = session.run("SHOW DATABASES")
                existing_dbs = [record["name"] for record in result]
                
                if self._database not in existing_dbs:
                    print(f"[Neo4jClient] Database '{self._database}' does not exist; creating it.")
                    session.run(f"CREATE DATABASE {self._database} IF NOT EXISTS")
                    print(f"[Neo4jClient] Database '{self._database}' created.")
        except Exception as e:
            # Fall back to the default database if database creation is unavailable.
            print(f"[Neo4jClient] Could not create database '{self._database}': {e}")
            print("[Neo4jClient] Falling back to the default database.")
            self._database = None
    
    def _get_session(self):
        """Get a database session."""
        if self._database:
            return self._driver.session(database=self._database)
        else:
            # Use the server default database.
            return self._driver.session()
    
    def save_memory(self, memory: MemoryItem) -> bool:
        """Save a memory node."""
        if self._use_mock:
            self._mock_store[memory.id] = memory
            return True
        
        try:
            with self._get_session() as session:
                session.run("""
                    MERGE (m:Memory {id: $id})
                    SET m.key = $key,
                        m.value = $value,
                        m.memory_type = $memory_type,
                        m.tags = $tags,
                        m.confidence = $confidence,
                        m.created_at = $created_at,
                        m.updated_at = $updated_at,
                        m.user_id = $user_id,
                        m.status = $status
                """, **memory.to_dict())
            return True
        except Exception as e:
            print(f"[Neo4jClient] Save failed: {e}")
            return False
    
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Get a memory node."""
        if self._use_mock:
            return self._mock_store.get(memory_id)
        
        try:
            with self._get_session() as session:
                result = session.run(
                    "MATCH (m:Memory {id: $id}) RETURN m",
                    id=memory_id
                )
                record = result.single()
                if record:
                    return MemoryItem.from_dict(dict(record["m"]))
        except Exception as e:
            print(f"[Neo4jClient] Query failed: {e}")
        return None
    
    def get_all_memories(self, user_id: str = None) -> List[MemoryItem]:
        """Get all memories."""
        if self._use_mock:
            memories = list(self._mock_store.values())
            if user_id:
                memories = [m for m in memories if m.user_id == user_id]
            return memories
        
        try:
            with self._get_session() as session:
                query = "MATCH (m:Memory) "
                if user_id:
                    query += "WHERE m.user_id = $user_id "
                query += "RETURN m"
                result = session.run(query, user_id=user_id)
                return [MemoryItem.from_dict(dict(r["m"])) for r in result]
        except Exception as e:
            print(f"[Neo4jClient] Query failed: {e}")
        return []
    
    def save_edge(self, edge: Edge) -> bool:
        """Save a relationship edge."""
        if self._use_mock:
            # Mock mode acts as a simple key-value store and does not add edges.
            print("[Neo4jClient] Warning: edge creation is disabled in mock mode.")
            return False
        
        try:
            with self._get_session() as session:
                session.run(f"""
                    MATCH (a:Memory {{id: $source_id}})
                    MATCH (b:Memory {{id: $target_id}})
                    MERGE (a)-[r:{edge.relation_type}]->(b)
                    SET r.weight = $weight
                """, source_id=edge.source_id, target_id=edge.target_id,
                    weight=edge.weight)
            return True
        except Exception as e:
            print(f"[Neo4jClient] Edge save failed: {e}")
            return False
    
    def get_related_memories(self, memory_id: str, 
                             relation_type: str = None) -> List[Tuple[MemoryItem, str]]:
        """Get related memories."""
        if self._use_mock:
            results = []
            for edge in self._mock_edges:
                if edge.source_id == memory_id:
                    if relation_type is None or edge.relation_type == relation_type:
                        mem = self._mock_store.get(edge.target_id)
                        if mem:
                            results.append((mem, edge.relation_type))
            return results
        
        try:
            with self._get_session() as session:
                query = """
                    MATCH (a:Memory {id: $id})-[r]->(b:Memory)
                    RETURN b, type(r) as rel_type
                """
                result = session.run(query, id=memory_id)
                return [(MemoryItem.from_dict(dict(r["b"])), r["rel_type"]) 
                        for r in result]
        except Exception as e:
            print(f"[Neo4jClient] Query failed: {e}")
        return []
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        if self._use_mock:
            if memory_id in self._mock_store:
                del self._mock_store[memory_id]
                self._mock_edges = [e for e in self._mock_edges 
                                    if e.source_id != memory_id and e.target_id != memory_id]
                return True
            return False
        
        try:
            with self._get_session() as session:
                session.run(
                    "MATCH (m:Memory {id: $id}) DETACH DELETE m",
                    id=memory_id
                )
            return True
        except Exception as e:
            print(f"[Neo4jClient] Delete failed: {e}")
            return False
    
    def close(self):
        """Close the connection."""
        if self._driver:
            self._driver.close()


# =============================================================================
# Milvus vector database client
# =============================================================================

class MilvusClient:
    """
    Milvus client for vector retrieval with optional user_id filtering.
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
        self._collection = None
        self._use_mock = True
        self._mock_vectors: Dict[str, Dict[str, Any]] = {}  # {memory_id: {"embedding": [...], "user_id": "..."}}
        self._embedding_client = EmbeddingClient()
        
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
            
            # Connect with URI plus username/password credentials.
            # Enable SSL when the URI uses https.
            use_secure = MILVUS_URI.startswith("https://")
            connections.connect(
                alias="default",
                uri=MILVUS_URI,
                user=MILVUS_USER,
                password=MILVUS_PASSWORD,
                secure=use_secure
            )
            
            # Create or open the collection.
            if not utility.has_collection(MILVUS_COLLECTION):
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),  # User ID field.
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
                ]
                schema = CollectionSchema(fields, description="Memory embeddings with user_id filter")
                self._collection = Collection(MILVUS_COLLECTION, schema)
                # Create a vector index.
                self._collection.create_index(
                    field_name="embedding",
                    index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
                )
                # Create a scalar index for faster user_id filtering.
                self._collection.create_index(
                    field_name="user_id",
                    index_params={"index_type": "INVERTED"}
                )
            else:
                self._collection = Collection(MILVUS_COLLECTION)
            
            self._collection.load()
            self._use_mock = False
            print(f"[MilvusClient] Connected to {MILVUS_URI}")
        except Exception as e:
            print(f"[MilvusClient] Connection failed; using in-memory storage: {e}")
        
        self._initialized = True
    
    def insert(self, memory_id: str, embedding: List[float], user_id: str = "default_user") -> bool:
        """
        Insert a vector.
        
        Args:
            memory_id: Memory ID.
            embedding: Vector.
            user_id: User ID.
            
        Returns:
            Whether insertion succeeded.
        """
        if self._use_mock:
            self._mock_vectors[memory_id] = {"embedding": embedding, "user_id": user_id}
            return True
        
        try:
            self._collection.insert([[memory_id], [user_id], [embedding]])
            self._collection.flush()
            return True
        except Exception as e:
            print(f"[MilvusClient] Insert failed: {e}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = 10, 
               user_id: str = None) -> List[Tuple[str, float]]:
        """
        Search vectors with optional user_id filtering.
        
        Args:
            query_embedding: Query vector.
            top_k: Number of results.
            user_id: Optional user ID filter.
            
        Returns:
            List of (memory_id, score) pairs.
        """
        if self._use_mock:
            # Mock implementation: cosine similarity with user_id filtering.
            results = []
            for mid, data in self._mock_vectors.items():
                # user_id filter
                if user_id is not None and data["user_id"] != user_id:
                    continue
                score = self._embedding_client.similarity(query_embedding, data["embedding"])
                results.append((mid, score))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        
        try:
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            # Build the filter expression.
            expr = None
            if user_id is not None:
                expr = f'user_id == "{user_id}"'
            
            results = self._collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,  # Apply user_id filtering.
                output_fields=["id", "user_id"]
            )
            return [(hit.id, hit.score) for hit in results[0]]
        except Exception as e:
            print(f"[MilvusClient] Search failed: {e}")
            return []
    
    def delete(self, memory_id: str) -> bool:
        """Delete a vector."""
        if self._use_mock:
            if memory_id in self._mock_vectors:
                del self._mock_vectors[memory_id]
                return True
            return False
        
        try:
            self._collection.delete(f'id == "{memory_id}"')
            return True
        except Exception as e:
            print(f"[MilvusClient] Delete failed: {e}")
            return False
    
    def delete_by_user(self, user_id: str) -> bool:
        """
        Delete all vectors for a specific user.
        
        Args:
            user_id: User ID.
            
        Returns:
            Whether deletion succeeded.
        """
        if self._use_mock:
            to_delete = [mid for mid, data in self._mock_vectors.items() if data["user_id"] == user_id]
            for mid in to_delete:
                del self._mock_vectors[mid]
            return True
        
        try:
            self._collection.delete(f'user_id == "{user_id}"')
            return True
        except Exception as e:
            print(f"[MilvusClient] Delete by user failed: {e}")
            return False


# =============================================================================
# Unified storage manager
# =============================================================================

class MemoryStore:
    """
    Unified memory store with synchronized Neo4j and Milvus IDs.
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
        self.neo4j = Neo4jClient()
        self.milvus = MilvusClient()
        self.embedding = EmbeddingClient()
        
        # Determine mock status.
        if self.neo4j._use_mock and self.milvus._use_mock:
            self.use_mock = True
            print("[MemoryStore] Running in mock mode with in-memory storage.")
        elif not self.neo4j._use_mock and not self.milvus._use_mock:
            self.use_mock = False
            print("[MemoryStore] Running in database-backed mode.")
        else:
            raise ValueError("Configuration error: Neo4j and Milvus must both be mock-backed or database-backed.")

        self._initialized = True
        print("[MemoryStore] Unified storage manager initialized.")
    
    def save(self, memory: MemoryItem, generate_embedding: bool = True) -> bool:
        """
        Save a memory to Neo4j and Milvus using the same ID.
        
        Args:
            memory: Memory item.
            generate_embedding: Whether to generate an embedding.
            
        Returns:
            Whether the save succeeded.
        """
        try:
            # 1. Generate an embedding if needed.
            if generate_embedding or memory.embedding is None:
                text = f"{memory.key} {memory.value}"
                memory.embedding = self.embedding.embed(text)
            
            # 2. Save structured data to Neo4j.
            neo4j_success = self.neo4j.save_memory(memory)
            
            # 3. Save vector data to Milvus with the same ID and user_id.
            milvus_success = self.milvus.insert(memory.id, memory.embedding, memory.user_id)
            
            if neo4j_success and milvus_success:
                if not self.use_mock:
                    # Avoid noisy success logs during training in mock mode.
                    print(f"[MemoryStore] Save succeeded: {memory.id} - {memory.key} (user: {memory.user_id})")
                return True
            else:
                print(f"[MemoryStore] Partial save failure: Neo4j={neo4j_success}, Milvus={milvus_success}")
                return False
                
        except Exception as e:
            print(f"[MemoryStore] Save exception: {e}")
            return False
    
    def save_batch(self, memories: List[MemoryItem], generate_embedding: bool = True) -> List[bool]:
        """Save memories in a batch."""
        results = []
        for memory in memories:
            success = self.save(memory, generate_embedding)
            results.append(success)
        return results
    
    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """Get a memory."""
        return self.neo4j.get_memory(memory_id)
    
    def get_all(self, user_id: str = None) -> List[MemoryItem]:
        """Get all memories."""
        return self.neo4j.get_all_memories(user_id)
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory from both Neo4j and Milvus."""
        neo4j_success = self.neo4j.delete_memory(memory_id)
        milvus_success = self.milvus.delete(memory_id)
        return neo4j_success and milvus_success
    
    def search_similar(self, query: str, top_k: int = 10, 
                       user_id: str = None) -> List[Tuple[MemoryItem, float]]:
        """
        Search for similar memories.
        
        Args:
            query: Query text.
            top_k: Number of results.
            user_id: User ID filter applied in Milvus.
            
        Returns:
            List of (memory, similarity score) pairs.
        """
        # 1. Generate the query vector.
        query_emb = self.embedding.embed(query)
        
        # 2. Search vectors with user_id filtering at the Milvus layer.
        # Request extra results because status filtering happens later.
        vector_results = self.milvus.search(query_emb, top_k=top_k * 2, user_id=user_id)
        
        # 3. Fetch memory details and filter by status.
        results = []
        for memory_id, score in vector_results:
            memory = self.neo4j.get_memory(memory_id)
            if memory:
                # Status filtering; user_id has already been handled in Milvus.
                if memory.status != MemoryStatus.ACTIVATED.value:
                    continue
                results.append((memory, score))
        
        return results[:top_k]
    
    def find_related_memories(self, memory: MemoryItem, 
                              top_k: int = 10) -> List[Tuple[MemoryItem, float]]:
        """
        Find existing memories related to a new memory for update decisions.
        
        Args:
            memory: Newly extracted memory.
            top_k: Number of results.
            
        Returns:
            List of (related memory, similarity) pairs.
        """
        query = f"{memory.key} {memory.value}"
        results = self.search_similar(query, top_k=top_k, user_id=memory.user_id)
        # Exclude the memory itself.
        return [(m, s) for m, s in results if m.id != memory.id]

    def to_list(self) -> List[Dict]:
        if not self.use_mock:
            raise RuntimeError("to_list is only available when use_mock=True")
            
        results = []
        # Access the Neo4jClient mock store, which contains MemoryItem objects.
        for mem in self.neo4j._mock_store.values():
            # Use to_dict so embeddings are replaced by compact placeholders.
            item_dict = mem.to_dict()
            results.append(item_dict)
        return results

    def from_list(self, data: List[Dict]) -> None:
        if not self.use_mock:
            raise RuntimeError("from_list is only available when use_mock=True")
            
        # Clear existing mock data.
        self.neo4j._mock_store.clear()
        self.milvus._mock_vectors.clear()
        
        for item_dict in data:
            # Rebuild MemoryItem objects; embeddings are regenerated below.
            mem = MemoryItem.from_dict(item_dict)
            
            # Save regenerates embeddings and writes to both backends.
            self.save(mem)

# =============================================================================
# Global singleton accessors
# =============================================================================

def get_memory_store() -> MemoryStore:
    """Get the unified memory store singleton."""
    return MemoryStore()


def get_llm_client() -> LLMClient:
    """Get the LLM client singleton."""
    return LLMClient()


def get_embedding_client() -> EmbeddingClient:
    """Get the embedding client singleton."""
    return EmbeddingClient()


def get_neo4j_client() -> Neo4jClient:
    """Get the Neo4j client singleton."""
    return Neo4jClient()


def get_milvus_client() -> MilvusClient:
    """Get the Milvus client singleton."""
    return MilvusClient()


# =============================================================================
# Utility functions
# =============================================================================

def generate_id() -> str:
    """Generate a unique ID."""
    import uuid
    return str(uuid.uuid4())


def current_timestamp() -> str:
    """Get the current timestamp."""
    return datetime.now().isoformat()


def format_conversation(messages: List[ConversationMessage]) -> str:
    """Format a conversation as text."""
    lines = []
    for msg in messages:
        if not msg.timestamp or msg.timestamp == "" or msg.timestamp == " ":
            lines.append(f"{msg.role}: {msg.content}")
        else:
            lines.append(f"{msg.role}: [{msg.timestamp}] {msg.content}")
    return "\n".join(lines)


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Common module smoke test")
    print("=" * 60)
    
    # Test the LLM client.
    llm = get_llm_client()
    print(f"LLM client: {llm.model}")
    
    # Test the embedding client.
    emb = get_embedding_client()
    vec = emb.embed("test text")
    print(f"Embedding dimension: {len(vec)}")
    
    # Test the Neo4j client.
    neo4j = get_neo4j_client()
    test_mem = MemoryItem(
        id=generate_id(),
        key="test memory",
        value="this is a test memory",
        memory_type="UserMemory",
        tags=["test"]
    )
    neo4j.save_memory(test_mem)
    print(f"Neo4j save succeeded: {test_mem.id}")
    
    # Test the Milvus client.
    milvus = get_milvus_client()
    milvus.insert(test_mem.id, vec)
    results = milvus.search(vec, top_k=1)
    print(f"Milvus search results: {results}")
    
    print("\nAll smoke tests passed.")
