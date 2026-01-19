# 🧠 LLM Memory

**Hierarchical Long-Term Memory for LLM Agents**

> Real memory, not chat history. A cognitive architecture with decay, consolidation, and intent-aware retrieval.

---

## 🎯 The Problem

Most "memory" systems for LLMs are just vector similarity search over conversation history. They lack:
- **Memory decay** - everything stays equally "fresh" forever
- **Consolidation** - experiences never become knowledge
- **Conflict resolution** - contradictory facts coexist silently
- **Intent-aware retrieval** - all queries are treated the same

## 💡 The Solution

A three-tier hierarchical memory system inspired by cognitive science:

```
┌─────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL LAYER                              │
│         (Intent-aware, task-scoped memory access)               │
└─────────────────────────────────────────────────────────────────┘
                              ▲
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  SHORT-TERM     │ │   EPISODIC      │ │   SEMANTIC      │
│  MEMORY (STM)   │ │   MEMORY        │ │   MEMORY        │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ • Working ctx   │ │ • Events        │ │ • Facts         │
│ • Current task  │ │ • Experiences   │ │ • Patterns      │
│ • Fast decay    │ │ • Temporal tags │ │ • Generalizations│
│ • High capacity │ │ • Medium decay  │ │ • Slow decay    │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────► CONSOLIDATION ◄───────────────┘
                   (STM → Episodic → Semantic)
```

## ✨ Features

- **Three-tier memory hierarchy**: Short-term → Episodic → Semantic
- **Memory decay**: Ebbinghaus forgetting curve with importance weighting
- **Automatic consolidation**: Memories are promoted and abstracted over time
- **Conflict detection & resolution**: Handle contradictory information gracefully
- **Intent-aware retrieval**: Different query types access different memory strategies
- **Scoped contexts**: Project, user, and global memory scopes

## 📦 Installation

```bash
pip install llm-memory
```

Or from source:

```bash
git clone https://github.com/llm-memory/llm-memory.git
cd llm-memory
pip install -e ".[dev]"
```

## 🚀 Quick Start

```python
from llm_memory import MemorySystem

# Initialize memory system
memory = MemorySystem(
    user_id="user_123",
    scope="project_abc"
)

# Observe conversations (auto-extracts and stores memories)
memory.observe("I prefer using Python for backend development", role="user")
memory.observe("I'll use Python for this project then", role="assistant")

# Recall relevant memories based on intent
context = memory.recall(
    query="What language should I use for the API?",
    intent="preference",
    limit=5
)

# Memory reflection (agent reviews its memories)
insights = memory.reflect(topic="user preferences")
```

## 🏗️ Architecture

### Memory Types

| Type | Purpose | Decay Rate | Example |
|------|---------|------------|---------|
| **Short-term** | Working context | Fast (minutes) | Current conversation buffer |
| **Episodic** | Event memories | Medium (days) | "User debugged auth issue on Monday" |
| **Semantic** | Facts & patterns | Slow (weeks) | "User prefers async Python" |

### Importance Scoring

Memories are scored for importance based on:
- **Emotional salience** (sentiment analysis)
- **Novelty** (how different from existing memories)
- **Relevance frequency** (how often retrieved)
- **Causal significance** (affects downstream events)
- **User feedback** (explicit importance markers)

### Consolidation Pipeline

```
STM → Episodic:
  Trigger: End of task/session OR importance threshold
  Transform: Raw context → Structured episode with temporal tags
  
Episodic → Semantic:
  Trigger: N similar episodes detected
  Transform: Specific events → General pattern/fact
```

## 📖 Documentation

- [Architecture Deep Dive](docs/architecture.md)
- [Cognitive Foundations](docs/cognitive_foundations.md)
- [API Reference](docs/api_reference.md)

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llm_memory --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

## 🗺️ Roadmap

- [x] Stage 1: Foundation & Memory Store
- [ ] Stage 2: Memory Encoding & Decay
- [ ] Stage 3: Consolidation Pipeline
- [ ] Stage 4: Conflict Resolution
- [ ] Stage 5: Intent-Aware Retrieval
- [ ] Stage 6: Agent Integration
- [ ] Stage 7: Evaluation & Tuning

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built with 🧠 for smarter AI agents**
