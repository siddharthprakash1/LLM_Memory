"""
Configuration management for LLM Memory system.

Provides centralized configuration for:
- Storage backends
- Decay parameters
- Consolidation thresholds
- Retrieval settings
- LLM providers
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DecayConfig(BaseModel):
    """Configuration for memory decay behavior."""

    # Base decay rates (lambda in the forgetting curve)
    stm_decay_rate: float = Field(
        default=0.1,
        description="Decay rate for short-term memory (higher = faster decay)",
        ge=0.0,
        le=1.0,
    )
    episodic_decay_rate: float = Field(
        default=0.01,
        description="Decay rate for episodic memory",
        ge=0.0,
        le=1.0,
    )
    semantic_decay_rate: float = Field(
        default=0.001,
        description="Decay rate for semantic memory (slower decay)",
        ge=0.0,
        le=1.0,
    )

    # Strength thresholds
    min_strength_threshold: float = Field(
        default=0.1,
        description="Minimum strength before memory is eligible for garbage collection",
        ge=0.0,
        le=1.0,
    )

    # Rehearsal boost (accessing memory increases strength)
    rehearsal_boost: float = Field(
        default=0.2,
        description="Strength boost when memory is accessed",
        ge=0.0,
        le=1.0,
    )


class ConsolidationConfig(BaseModel):
    """Configuration for memory consolidation pipeline."""

    # STM → Episodic promotion
    stm_to_episodic_threshold: float = Field(
        default=0.6,
        description="Importance threshold for promoting STM to episodic",
        ge=0.0,
        le=1.0,
    )
    stm_max_age_seconds: int = Field(
        default=3600,  # 1 hour
        description="Maximum age of STM items before forced consolidation",
    )

    # Episodic → Semantic promotion
    episodic_similarity_threshold: float = Field(
        default=0.85,
        description="Similarity threshold for merging episodes into semantic memory",
        ge=0.0,
        le=1.0,
    )
    min_episodes_for_semantic: int = Field(
        default=3,
        description="Minimum similar episodes required to form semantic memory",
        ge=2,
    )

    # Background consolidation
    consolidation_interval_seconds: int = Field(
        default=300,  # 5 minutes
        description="Interval between consolidation runs",
    )


class StorageConfig(BaseModel):
    """Configuration for storage backends."""

    # SQLite settings
    sqlite_path: Path = Field(
        default=Path("./data/memory.db"),
        description="Path to SQLite database file",
    )

    # Vector store settings
    vector_store_type: Literal["chroma", "qdrant"] = Field(
        default="chroma",
        description="Vector store backend type",
    )
    chroma_persist_directory: Path = Field(
        default=Path("./data/chroma"),
        description="Directory for ChromaDB persistence",
    )
    qdrant_url: str | None = Field(
        default=None,
        description="Qdrant server URL (if using Qdrant)",
    )

    # Collection names
    stm_collection: str = Field(
        default="short_term_memory",
        description="Collection name for STM vectors",
    )
    episodic_collection: str = Field(
        default="episodic_memory",
        description="Collection name for episodic vectors",
    )
    semantic_collection: str = Field(
        default="semantic_memory",
        description="Collection name for semantic vectors",
    )


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    provider: Literal["openai", "ollama", "sentence-transformers"] = Field(
        default="ollama",
        description="Embedding provider (openai, ollama, or sentence-transformers)",
    )
    model: str = Field(
        default="nomic-embed-text",
        description="Embedding model name (e.g., nomic-embed-text for Ollama)",
    )
    dimensions: int = Field(
        default=768,
        description="Embedding vector dimensions (768 for nomic-embed-text)",
    )
    batch_size: int = Field(
        default=100,
        description="Batch size for embedding requests",
    )
    
    # Ollama-specific settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL",
    )


class LLMConfig(BaseModel):
    """Configuration for LLM operations (summarization, conflict resolution)."""

    provider: Literal["openai", "anthropic", "ollama"] = Field(
        default="ollama",
        description="LLM provider for memory operations",
    )
    model: str = Field(
        default="llama3.2",
        description="Model for memory operations (e.g., llama3.2 for Ollama)",
    )
    temperature: float = Field(
        default=0.3,
        description="Temperature for LLM responses",
        ge=0.0,
        le=2.0,
    )
    max_tokens: int = Field(
        default=500,
        description="Maximum tokens for LLM responses",
    )
    
    # Ollama-specific settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL",
    )


class RetrievalConfig(BaseModel):
    """Configuration for memory retrieval."""

    default_limit: int = Field(
        default=10,
        description="Default number of memories to retrieve",
    )
    max_limit: int = Field(
        default=50,
        description="Maximum number of memories to retrieve",
    )

    # Retrieval weights by intent
    intent_weights: dict[str, dict[str, float]] = Field(
        default={
            "factual": {"semantic": 0.7, "episodic": 0.2, "stm": 0.1},
            "procedural": {"semantic": 0.3, "episodic": 0.6, "stm": 0.1},
            "contextual": {"semantic": 0.1, "episodic": 0.6, "stm": 0.3},
            "preference": {"semantic": 0.8, "episodic": 0.15, "stm": 0.05},
            "debugging": {"semantic": 0.1, "episodic": 0.3, "stm": 0.6},
        },
        description="Weight distribution for different retrieval intents",
    )

    # Reranking
    enable_reranking: bool = Field(
        default=True,
        description="Enable LLM-based reranking of results",
    )


class MemoryConfig(BaseModel):
    """Master configuration for the LLM Memory system."""

    # Sub-configurations
    decay: DecayConfig = Field(default_factory=DecayConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    # Global settings
    user_id: str | None = Field(
        default=None,
        description="Default user ID for memory scoping",
    )
    default_scope: str = Field(
        default="global",
        description="Default scope for memories (global, project, session)",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug logging",
    )

    @classmethod
    def from_file(cls, path: Path) -> "MemoryConfig":
        """Load configuration from a JSON or YAML file."""
        import json

        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            return cls.model_validate(data)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

    def to_file(self, path: Path) -> None:
        """Save configuration to a JSON file."""
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2, default=str)
