# Claude Instructions — LLM Memory Project

## Project Overview

LLM Memory is a **local-first, fact-structured long-term memory system** for LLM agents. It extracts structured facts from conversations at ingest time (not retrieval time), tracks temporal state, resolves conflicts, and retrieves with multi-angle search + multi-hop reasoning. It runs entirely locally using Ollama for LLM inference and sentence-transformers for embeddings.

## Architecture

### Active Memory Stack: V5 (primary), V4 (reference/fallback)

| Component | V5 Location | V4 Reference |
|---|---|---|
| Memory Store (orchestrator) | `llm_memory/memory_v5/memory_store_v5.py` | `llm_memory/memory_v4/memory_store.py` |
| Graph Store | `llm_memory/memory_v5/graph_store.py` | — (facts are flat dicts in V4) |
| Retrieval | `llm_memory/memory_v5/retrieval_v5.py` | `llm_memory/memory_v4/retrieval.py` |
| Embedder + Reranker | `llm_memory/memory_v5/embedder.py` | — (new in V5) |
| Temporal Tracker | `llm_memory/memory_v5/temporal_v5.py` | `llm_memory/memory_v4/temporal_state.py` |
| Tiered Memory | `llm_memory/memory_v5/tiered_memory.py` | — (V5 only) |
| Memory Manager | `llm_memory/memory_v5/memory_manager.py` | `llm_memory/memory_v4/conflict_resolver.py` |
| Agents (LangGraph) | `llm_memory/agents_v5/` | `llm_memory/agents_v4/` |
| Web UI | `llm_memory/agents_v4/web_ui.py` | — |

### Key Design Principles

1. **Facts-first**: Extract structured facts at ingest time using LLM, not raw text storage
2. **Episode fallback**: Always keep raw conversation text alongside facts for safety
3. **Embedding-based retrieval**: Use `sentence-transformers/all-MiniLM-L6-v2` (384d, local, no API) as primary retrieval signal
4. **Cross-encoder reranking**: Use `cross-encoder/ms-marco-MiniLM-L6-v2` for post-retrieval reranking
5. **Graph augments, not replaces**: Graph traversal enriches retrieval for multi-hop — flat fact index + embeddings remain primary
6. **Local-first**: Ollama for LLM, sentence-transformers for embeddings — no cloud APIs required
7. **Inspectable**: UI shows facts, graph, timeline in real time

## Coding Conventions

- **Python 3.11+**, type hints everywhere
- **Pydantic** for config, **dataclasses** for data objects
- **SQLite** for persistence, in-memory dicts for fast access
- **No unnecessary print statements** — use `logging` module if needed for debug
- Imports: stdlib → third-party → local, separated by blank lines
- Docstrings: Google-style, with Args/Returns sections
- Test with `pytest` under `tests/`

## Model Choices

| Purpose | Model | Dimension | Source |
|---|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` | 384 | sentence-transformers (local, no API) |
| Reranker | `ms-marco-MiniLM-L6-v2` | — | cross-encoder (local, no API) |
| LLM (default) | `qwen2.5:32b` | — | Ollama (local) |
| LLM (fast) | `qwen2.5:7b` | — | Ollama (local) |

## Key Benchmarks

- **LOCOMO**: Primary benchmark — tests single-hop, multi-hop, temporal, adversarial
- **LongMemEval**: Secondary benchmark — tests long-context memory
- Reports go to `benchmarks/reports/`

## File Naming

- V5 files: `*_v5.py` or under `memory_v5/`, `agents_v5/`
- V4 files: `*_v4.py` or under `memory_v4/`, `agents_v4/` (reference, do not modify)
- Benchmark scripts: under `benchmarks/`
- Generated data dirs: gitignored (`*_memory/`, `*.log`, `*.db`)

## What NOT to Do

- Don't add `print()` debug statements — use logging or remove before commit
- Don't add API key requirements for core functionality — keep local-first
- Don't modify V4 files — they are the reference implementation
- Don't store generated memory DBs or log files in git
