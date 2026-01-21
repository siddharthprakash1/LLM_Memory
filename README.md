# 🧠 LLM Memory

<div align="center">

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)

**Hierarchical Long-Term Memory for LLM Agents**

*Real memory with decay, consolidation, conflict resolution, and intent-aware retrieval*

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Benchmarks](#-benchmarks) • [API Reference](#-api-reference)

</div>

---

## 🎯 What is LLM Memory?

LLM Memory is a **production-grade cognitive memory system** for AI agents. Unlike simple context windows or basic RAG, it implements a biologically-inspired memory architecture with:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         🧠 LLM MEMORY SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   📥 INPUT                    🔄 PROCESSING              📤 OUTPUT      │
│   ─────────                   ──────────────             ────────       │
│                                                                         │
│   "Remember    ──────►    ┌──────────────┐    ──────►   Structured     │
│    my name                │   Encoding   │              Memory         │
│    is John"               │  (Embedding) │              Storage        │
│                           └──────────────┘                             │
│                                  │                                      │
│                                  ▼                                      │
│                           ┌──────────────┐                             │
│                           │   Memory     │                             │
│   "What's     ◄──────     │   Decay &    │    ◄──────   Retrieval     │
│    my name?"              │ Consolidation│              with RAG       │
│                           └──────────────┘                             │
│                                                                         │
│   "Your name              ┌──────────────┐              Vector         │
│    is John"   ◄──────     │   Conflict   │    ◄──────   Search        │
│                           │  Resolution  │                             │
│                           └──────────────┘                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## ✨ Features

### Memory Types

| Type | Description | Use Case | Retention |
|------|-------------|----------|-----------|
| 🔴 **Short-Term** | Recent conversation context | Current session | Minutes to hours |
| 🟡 **Episodic** | Specific events & experiences | "What happened when..." | Days to weeks |
| 🟢 **Semantic** | Facts, concepts, relationships | "What is X?" | Long-term |

### Core Capabilities

```
┌────────────────────────────────────────────────────────────────────┐
│                        FEATURE MATRIX                              │
├─────────────────────┬───────────────┬─────────────────────────────┤
│ Feature             │ Status        │ Description                 │
├─────────────────────┼───────────────┼─────────────────────────────┤
│ Vector Search       │ ✅ Production │ ChromaDB + HNSW indexing    │
│ RAG Pipeline        │ ✅ Production │ LLM answer synthesis        │
│ Memory Decay        │ ✅ Production │ Ebbinghaus forgetting curve │
│ Consolidation       │ ✅ Production │ STM → Episodic → Semantic   │
│ Conflict Resolution │ ✅ Production │ 6 detection + 8 strategies  │
│ Multi-hop Reasoning │ ✅ Production │ Iterative retrieval         │
│ Temporal Logic      │ ✅ Production │ Time-aware scoring          │
│ Intent Classification│ ✅ Production │ Query understanding        │
│ LangChain Support   │ ✅ Production │ Full integration            │
│ Ollama Support      │ ✅ Production │ Local LLM inference         │
└─────────────────────┴───────────────┴─────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) (for local LLM)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/siddharthprakash1/LLM_Memory.git
cd LLM_Memory

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev,langchain,agent,ui]"

# Pull required Ollama models
ollama pull gemma3:27b        # Main LLM (or any model you prefer)
ollama pull nomic-embed-text  # Embedding model
```

### Docker (Coming Soon)

```bash
docker pull llmmemory/llm-memory:latest
docker run -p 8000:8000 llmmemory/llm-memory
```

---

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from llm_memory import MemorySystem, MemoryConfig

async def main():
    # Initialize memory system
    memory = MemorySystem()
    await memory.initialize()
    await memory.start()
    
    # Store memories
    await memory.remember("My name is Alice", user_id="user_1")
    await memory.remember("I work at OpenAI as a researcher", user_id="user_1")
    await memory.remember("My favorite color is purple", user_id="user_1")
    
    # Recall memories
    results = await memory.recall("What is my name?", user_id="user_1")
    print(results)  # Returns relevant memories about Alice
    
    # Cleanup
    await memory.stop()

asyncio.run(main())
```

### With RAG Pipeline

```python
from llm_memory.retrieval import RAGPipeline, create_rag_pipeline

async def main():
    # Create RAG pipeline
    pipeline = await create_rag_pipeline(
        persist_directory="./memory_store",
        embed_func=your_embed_function,
        llm_func=your_llm_function,
    )
    
    # Get natural language answers
    result = await pipeline.answer("What does Alice do for work?")
    
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Sources: {len(result.sources)}")

asyncio.run(main())
```

### CLI Agent

```bash
# Run the memory-powered chat agent
python -m llm_memory.agent.cli

# Commands:
#   /help     - Show available commands
#   /memory   - View memory statistics
#   /remember - Store a memory
#   /recall   - Search memories
#   /new      - Start new session
#   /quit     - Exit
```

### Web UI

```bash
# Launch Gradio interface
python -m llm_memory.agent.web_ui

# Open http://localhost:7860 in your browser
```

---

## 🏗 Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           LLM MEMORY ARCHITECTURE                            │
└──────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │   User /    │
                              │   Agent     │
                              └──────┬──────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
            ┌───────────┐    ┌───────────┐    ┌───────────┐
            │  Remember │    │   Recall  │    │   Forget  │
            │    API    │    │    API    │    │    API    │
            └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
                  │                │                │
                  └────────────────┼────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          MEMORY SYSTEM ORCHESTRATOR                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         Processing Pipeline                              │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │ │
│  │  │ Encoding │→ │  Intent  │→ │ Conflict │→ │ Storage  │→ │  Index   │  │ │
│  │  │          │  │ Classify │  │  Check   │  │          │  │          │  │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         Retrieval Pipeline                               │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │ │
│  │  │  Query   │→ │  Vector  │→ │ Temporal │→ │ Multi-   │→ │   RAG    │  │ │
│  │  │  Embed   │  │  Search  │  │ Scoring  │  │   Hop    │  │ Synthesis│  │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                       Background Processes                               │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │ │
│  │  │    Memory    │  │  Garbage     │  │   Decay      │                   │ │
│  │  │ Consolidation│  │  Collection  │  │   Updates    │                   │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
            ┌───────────┐  ┌───────────┐  ┌───────────┐
            │  SQLite   │  │ ChromaDB  │  │   Event   │
            │ (Metadata)│  │ (Vectors) │  │   Hooks   │
            └───────────┘  └───────────┘  └───────────┘
```

### Memory Hierarchy

```
                    ┌─────────────────────────────────────┐
                    │         WORKING MEMORY              │
                    │    (Active conversation context)    │
                    │         Capacity: ~10 items         │
                    │         Duration: Seconds           │
                    └──────────────────┬──────────────────┘
                                       │
                              ┌────────┴────────┐
                              │   ATTENTION &   │
                              │   REHEARSAL     │
                              └────────┬────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │         SHORT-TERM MEMORY           │
                    │      (Recent interactions)          │
                    │      Capacity: ~100 items           │
                    │      Duration: Minutes-Hours        │
                    │      Decay: Fast exponential        │
                    └──────────────────┬──────────────────┘
                                       │
                              ┌────────┴────────┐
                              │  CONSOLIDATION  │
                              │  (Sleep-like)   │
                              └────────┬────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            │                                                     │
            ▼                                                     ▼
┌───────────────────────────┐               ┌───────────────────────────┐
│     EPISODIC MEMORY       │               │     SEMANTIC MEMORY       │
│  (Events & Experiences)   │               │   (Facts & Concepts)      │
│                           │               │                           │
│  • "What happened when"   │               │  • "What is X"            │
│  • Contextual details     │               │  • General knowledge      │
│  • Temporal ordering      │               │  • Relationships          │
│  • Emotional tags         │               │  • Abstract concepts      │
│                           │               │                           │
│  Duration: Days-Weeks     │   ─────────►  │  Duration: Long-term      │
│  Decay: Moderate          │  Abstraction  │  Decay: Very slow         │
└───────────────────────────┘               └───────────────────────────┘
```

### Memory Decay (Ebbinghaus Curve)

```
Strength
   │
1.0├────●
   │     ╲
   │      ╲
0.8├       ╲
   │        ╲
   │         ╲                    ◆ With rehearsal
0.6├          ╲               ◆
   │           ╲          ◆
   │            ╲     ◆
0.4├             ╲◆
   │              ╲
   │               ╲
0.2├                ╲──────────────────  Without rehearsal
   │                 ╲
   │                  ╲
0.0├───────────────────╲────────────────────────────►
   0    1    2    3    4    5    6    7   Time (days)

   Formula: S(t) = S₀ × e^(-λt/importance)
   
   Where:
   • S₀ = Initial strength
   • λ = Decay rate (configurable)
   • t = Time since last access
   • importance = Memory importance score (slows decay)
```

### Conflict Resolution

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONFLICT DETECTION & RESOLUTION                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CONFLICT TYPES                      RESOLUTION STRATEGIES          │
│  ──────────────                      ─────────────────────          │
│                                                                     │
│  ┌─────────────────┐                 ┌─────────────────┐           │
│  │ Direct          │   ──────────►   │ Recency         │           │
│  │ Contradiction   │                 │ (newest wins)   │           │
│  └─────────────────┘                 └─────────────────┘           │
│                                                                     │
│  ┌─────────────────┐                 ┌─────────────────┐           │
│  │ Temporal        │   ──────────►   │ Confidence      │           │
│  │ Outdated        │                 │ (highest wins)  │           │
│  └─────────────────┘                 └─────────────────┘           │
│                                                                     │
│  ┌─────────────────┐                 ┌─────────────────┐           │
│  │ Source          │   ──────────►   │ Source          │           │
│  │ Disagreement    │                 │ Reliability     │           │
│  └─────────────────┘                 └─────────────────┘           │
│                                                                     │
│  ┌─────────────────┐                 ┌─────────────────┐           │
│  │ Preference      │   ──────────►   │ Merge           │           │
│  │ Conflict        │                 │ (combine both)  │           │
│  └─────────────────┘                 └─────────────────┘           │
│                                                                     │
│  ┌─────────────────┐                 ┌─────────────────┐           │
│  │ Fact            │   ──────────►   │ User-Guided     │           │
│  │ Inconsistency   │                 │ (ask user)      │           │
│  └─────────────────┘                 └─────────────────┘           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmarks

### Test Configuration

| Parameter | Value |
|-----------|-------|
| Model | Gemma 3 27B (Ollama) |
| Embedding | nomic-embed-text |
| Samples | 15-30 per scenario |
| Runs | 2 per scenario |

### Results

```
┌────────────────────────────────────────────────────────────────────────┐
│                         BENCHMARK RESULTS                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ACCURACY (Contains Match)                                             │
│  ─────────────────────────                                             │
│                                                                        │
│  Single-hop    ████████████████████████████████████████████  100%     │
│  Multi-hop     ████████████████████████████████████████████  100%     │
│  Temporal      ████████████████████████████████████░░░░░░░░   90%     │
│  Conflict      ██████████████████████████░░░░░░░░░░░░░░░░░░   67%     │
│                                                                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  LATENCY (p95)                                                         │
│  ─────────────                                                         │
│                                                                        │
│  Single-hop    ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░  4.8s               │
│  Multi-hop     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░  9.0s               │
│  Temporal      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░  6.2s               │
│  Conflict      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░  7.9s               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Comparison with Other Systems

| System | Single-hop | Multi-hop | Temporal | Conflict | Notes |
|--------|------------|-----------|----------|----------|-------|
| **LLM Memory** | **100%** | **100%** | **90%** | **67%** | Our system |
| Full History | 100% | 100% | 100% | 100% | Doesn't scale |
| Simple RAG | 70% | 40% | 50% | 30% | No temporal logic |
| Mem0 (reported) | 72.5% | - | - | - | Production system |
| MemGPT | 75% | 60% | - | - | Hierarchical |

### Run Your Own Benchmark

```bash
# Full benchmark suite
python -m benchmarks.benchmark_memory \
    --scenarios single_hop,multi_hop,temporal,conflict \
    --samples 30 \
    --runs 3 \
    --model gemma3:27b

# Quick test
python -m benchmarks.benchmark_memory \
    --scenarios single_hop \
    --samples 10 \
    --runs 1
```

---

## 📚 API Reference

### Core Classes

#### `MemorySystem`

The main orchestrator for all memory operations.

```python
from llm_memory import MemorySystem, MemoryConfig

# Initialize
config = MemoryConfig(
    llm=LLMConfig(provider="ollama", model="gemma3:27b"),
    embedding=EmbeddingConfig(provider="ollama", model="nomic-embed-text"),
)
memory = MemorySystem(config)
await memory.initialize()
await memory.start()

# Store
memory_obj = await memory.remember(
    content="User prefers dark mode",
    user_id="user_123",
    tags=["preference", "ui"],
)

# Retrieve
results = await memory.recall(
    query="What theme does the user prefer?",
    user_id="user_123",
    limit=5,
)

# Stats
stats = memory.get_statistics()
```

#### `RAGPipeline`

Production RAG with LLM synthesis.

```python
from llm_memory.retrieval import RAGPipeline, RAGConfig

config = RAGConfig(
    top_k=10,
    enable_temporal_scoring=True,
    enable_multi_hop=True,
    temporal_weight=0.3,
)

pipeline = RAGPipeline(
    vector_engine=vector_engine,
    embed_func=embedder.embed,
    llm_func=llm.generate,
    config=config,
)

result = await pipeline.answer("What is Alice's job?")
# result.answer = "Alice works at OpenAI as a researcher [1]."
# result.confidence = 0.85
# result.quality = AnswerQuality.HIGH
```

#### `VectorSearchEngine`

ChromaDB-backed vector search.

```python
from llm_memory.retrieval import VectorSearchEngine, VectorSearchConfig

config = VectorSearchConfig(
    collection_name="my_memories",
    hnsw_space="cosine",
    similarity_threshold=0.5,
)

engine = VectorSearchEngine(config)
await engine.initialize()

# Add memories
await engine.add_memory(memory, embedding)

# Search
results = await engine.search(query_embedding, k=10)
# or hybrid search
results = await engine.hybrid_search(query_embedding, query_text, k=10)
```

#### `MultiHopReasoner`

Complex query decomposition and iterative retrieval.

```python
from llm_memory.retrieval import MultiHopReasoner, MultiHopConfig

config = MultiHopConfig(
    max_hops=5,
    min_confidence=0.3,
    memories_per_hop=3,
)

reasoner = MultiHopReasoner(
    retrieve_func=my_retrieve_func,
    llm_func=my_llm_func,
    config=config,
)

path = await reasoner.reason(
    "What is the capital of the country where my friend lives?"
)
# path.hops = [ReasoningHop(...), ...]
# path.final_answer = "Paris"
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_LLM_MODEL=gemma3:27b
DEFAULT_EMBEDDING_MODEL=nomic-embed-text

# Storage
MEMORY_PERSIST_DIR=./memory_data
CHROMA_PERSIST_DIR=./chroma_data

# Memory Settings
MEMORY_DECAY_RATE=0.1
CONSOLIDATION_INTERVAL=300  # seconds
```

### Configuration Options

```python
from llm_memory.config import (
    MemoryConfig,
    LLMConfig,
    EmbeddingConfig,
    StorageConfig,
    DecayConfig,
)

config = MemoryConfig(
    # LLM settings
    llm=LLMConfig(
        provider="ollama",           # ollama, openai, anthropic
        model="gemma3:27b",
        temperature=0.7,
        max_tokens=1000,
    ),
    
    # Embedding settings
    embedding=EmbeddingConfig(
        provider="ollama",
        model="nomic-embed-text",
        dimensions=768,
    ),
    
    # Storage settings
    storage=StorageConfig(
        backend="sqlite",
        path="./memory.db",
    ),
    
    # Decay settings
    decay=DecayConfig(
        function="ebbinghaus",       # ebbinghaus, power_law, linear
        rate=0.1,
        importance_factor=0.5,
    ),
)
```

---

## 🔧 Advanced Usage

### Event Hooks

```python
from llm_memory.api import EventHooks

hooks = EventHooks()

@hooks.on_store
async def log_store(memory):
    print(f"Stored: {memory.id}")

@hooks.on_recall
async def log_recall(query, results):
    print(f"Query: {query}, Found: {len(results)}")

@hooks.on_conflict
async def handle_conflict(old, new, conflict_type):
    print(f"Conflict detected: {conflict_type}")
    return "keep_new"  # Resolution strategy

memory = MemorySystem(config, hooks=hooks)
```

### LangChain Integration

```python
from llm_memory.api.integrations import LLMMemory
from langchain.chains import ConversationChain

# Create LangChain-compatible memory
memory = LLMMemory(
    memory_system=my_memory_system,
    session_id="session_123",
)

# Use in chain
chain = ConversationChain(
    llm=my_llm,
    memory=memory,
)

response = chain.run("What's my name?")
```

### Custom Memory Types

```python
from llm_memory.models import BaseMemory, MemoryType
from pydantic import Field

class TaskMemory(BaseMemory):
    """Custom memory type for tasks."""
    
    memory_type: MemoryType = MemoryType.SEMANTIC
    
    # Custom fields
    priority: int = Field(default=1, ge=1, le=5)
    due_date: datetime | None = None
    status: str = "pending"
    
    def get_summary(self) -> str:
        return f"[P{self.priority}] {self.content} ({self.status})"
```

---

## 📁 Project Structure

```
llm_memory/
├── __init__.py
├── config.py                 # Configuration classes
├── models/                   # Data models
│   ├── base.py              # BaseMemory, ImportanceFactors
│   ├── short_term.py        # ShortTermMemory, WorkingContext
│   ├── episodic.py          # EpisodicMemory, Episode
│   └── semantic.py          # SemanticMemory, Fact, Concept
├── storage/                  # Storage backends
│   ├── base.py              # Abstract base
│   ├── sqlite.py            # SQLite implementation
│   └── vector.py            # ChromaDB wrapper
├── encoding/                 # Memory encoding
│   ├── embedder.py          # Embedding generation
│   └── summarizer.py        # LLM summarization
├── decay/                    # Memory decay
│   ├── functions.py         # Decay algorithms
│   └── scheduler.py         # Background decay
├── consolidation/            # Memory consolidation
│   ├── pipeline.py          # Consolidation logic
│   └── merger.py            # Memory merging
├── conflict/                 # Conflict resolution
│   ├── detector.py          # Conflict detection
│   └── resolver.py          # Resolution strategies
├── retrieval/                # Memory retrieval
│   ├── intent.py            # Intent classification
│   ├── searcher.py          # Memory search
│   ├── ranker.py            # Result ranking
│   ├── vector_search.py     # ChromaDB search
│   ├── temporal.py          # Time-aware scoring
│   ├── multi_hop.py         # Multi-hop reasoning
│   └── rag_pipeline.py      # Full RAG pipeline
├── api/                      # External APIs
│   ├── memory_api.py        # Programmatic API
│   ├── memory_system.py     # Main orchestrator
│   ├── hooks.py             # Event hooks
│   └── integrations/        # Third-party integrations
│       └── langchain.py
└── agent/                    # Agent implementations
    ├── cli.py               # CLI interface
    ├── web_ui.py            # Gradio UI
    ├── memory_agent.py      # LangGraph agent
    └── tools.py             # Agent tools

benchmarks/                   # Benchmarking suite
├── benchmark_memory.py      # Main benchmark script
├── scenarios.py             # Test scenarios
├── metrics.py               # Evaluation metrics
├── runner.py                # Benchmark runner
└── reports/                 # Generated reports
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/LLM_Memory.git
cd LLM_Memory

# Create branch
git checkout -b feature/your-feature

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=llm_memory

# Run linting
ruff check llm_memory/
black llm_memory/

# Submit PR
git push origin feature/your-feature
```

### Development Guidelines

1. **Tests**: Add tests for new features
2. **Types**: Use type hints
3. **Docs**: Update docstrings and README
4. **Style**: Follow black + ruff formatting

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

This project builds upon research from:

- **MemoryAgentBench** - Evaluation framework
- **Mem0** - Production memory patterns
- **MemGPT** - Hierarchical memory concepts
- **TiMem** - Temporal memory reasoning
- **Soar** - Cognitive architecture inspiration

---

<div align="center">

**[⬆ Back to Top](#-llm-memory)**

Made with ❤️ by the LLM Memory Team

</div>
