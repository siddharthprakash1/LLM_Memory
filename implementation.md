# V5 Performance Overhaul — Implementation Plan

## Goal

Make Memory V5 **beat V4** on LOCOMO benchmarks and **compete with mem0** by fixing 5 root causes of V5's underperformance, adding embedding-based retrieval, and cleaning up the codebase.

## Root Cause Analysis

| # | Problem | Impact | Fix |
|---|---|---|---|
| 1 | No embedding-based retrieval — word-overlap only | Low recall on semantically related queries | Add `sentence-transformers` + cosine similarity as primary retrieval |
| 2 | STM capacity=20, rehearsal_count≥2 for LTM promotion | Catastrophic memory loss on long conversations | Auto-promote to LTM, remove rehearsal gate |
| 3 | No episode/raw text storage — extraction failures lose info | Permanent information loss | Restore episode table + keyword search fallback |
| 4 | Weak extraction prompt (generic, no typed facts) | Fewer and less structured facts extracted | Port V4's detailed EXTRACTION_PROMPT |
| 5 | Dead-weight components (ReflectiveManager=no-op, MemoryManager adds LLM calls) | Latency with zero accuracy gain | Remove reflective, default manager to rule-based |

## Implementation Checklist

### Phase 1: Infrastructure

- [x] Create `claude.md` (project instructions for AI agents)
- [x] Create `implementation.md` (this file)
- [ ] Create `llm_memory/memory_v5/embedder.py`
  - `MemoryEmbedder` class wrapping `SentenceTransformer("all-MiniLM-L6-v2")`
  - `encode(texts) -> np.ndarray` with lazy model loading
  - `similarity(query_emb, corpus_embs) -> np.ndarray` using cosine similarity
  - `RetrievalReranker` class wrapping `CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")`
  - `rerank(query, candidates, top_k) -> List[RetrievalResult]`
- [ ] Update `requirements.txt`: add `sentence-transformers>=2.2.0`
- [ ] Update `pyproject.toml`: add `sentence-transformers>=2.2.0` to dependencies

### Phase 2: Fix Core V5 Regressions

- [ ] **Fix `Triplet.as_text()`** in `graph_store.py`
  - Include `source_date`, `confidence`, `temporal_scope` in text representation
  - Before: `"Caroline likes hiking"`
  - After: `"Caroline likes hiking [2024-01-15, confidence: 0.9]"`

- [ ] **Wire embeddings into `graph_store.py`**
  - Replace no-op `_compute_embedding()` with `MemoryEmbedder.encode()`
  - Embed entities at `add_entity()` time
  - Embed triplet text at `add_triplet()` time, store in `triplet_embeddings` dict
  - Add `semantic_search_triplets(query_embedding, top_k)` method

- [ ] **Fix STM bottleneck** in `tiered_memory.py`
  - In `get_promotable_to_ltm()`: change from `importance >= 0.6 AND rehearsal_count >= 2` to `importance >= 0.4` only
  - In `ShortTermMemory.add()`: after adding, immediately call promotion check
  - Increase STM capacity from 20 to 200 as safety net

- [ ] **Restore episode storage** in `memory_store_v5.py`
  - Add `Episode` dataclass (port from V4: speaker, original_text, normalized_text, date, session_id, fact_ids)
  - Add `episodes: Dict[str, Episode]` to `MemoryStoreV5`
  - Add `episodes` SQLite table in `_init_db()`
  - Store episode in `add_conversation_turn()` after extraction
  - Add `get_episodes_for_speaker()` and `search_episodes()` methods

- [ ] **Port V4 extraction prompt** to `memory_store_v5.py`
  - Replace generic `_extract_entities_and_relations()` prompt with V4's `EXTRACTION_PROMPT`
  - Adapt output parsing to extract both V4-style facts AND V5-style entities/relations
  - Remove `[DEBUG] Extracting` print statements

### Phase 3: Advanced Retrieval

- [ ] **Add semantic retrieval** to `retrieval_v5.py`
  - New `_retrieve_semantic(query, top_k)` method using `MemoryEmbedder`
  - Embed query → cosine similarity against stored triplet embeddings
  - Wire into `retrieve()` alongside graph and tiered search

- [ ] **Integrate cross-encoder reranker** into `retrieval_v5.py`
  - In `_rerank_results()`: use `RetrievalReranker.rerank()` as primary ranking signal
  - Keep source-type bonuses (graph +0.05, CoE +0.1) as tiebreakers
  - Fall back to word-overlap scoring if reranker model unavailable

- [ ] **Add episode search** to `retrieval_v5.py`
  - Port `_search_episodes()` from V4's retrieval.py
  - Include episode results in `build_context()` under "RELEVANT CONVERSATIONS:" section
  - If < 5 facts found, also fetch recent episodes by speaker (V4 pattern)

### Phase 4: Remove Dead Weight

- [ ] **Remove ReflectiveManager** from `memory_store_v5.py`
  - Remove Stage 5 (`self.reflective.prospective.add_utterance(...)`) from `add_conversation_turn()`
  - Remove `use_reflection` branch from `query()` — results are never merged
  - Keep `reflective.py` file but remove its usage from the pipeline
  - Remove `self.reflective` init from `_init_components()`

- [ ] **Default MemoryManager to rule-based** in `memory_manager.py`
  - Change `use_llm` default from `True` to `False`
  - Rule-based decisions (Jaccard similarity + contradiction patterns) are sufficient and much faster

### Phase 5: Cleanup

- [ ] **Delete root-level log files**
  - `benchmark_28dc39ac_parallel.log`
  - `benchmark_28dc39ac.log`
  - `benchmark_output.log`
  - `benchmark_28dc39ac_parallel_100.log`
  - `benchmark_28dc39ac_final.log`
  - `benchmark_28dc39ac_optimized.log`

- [ ] **Delete debug scripts**
  - `debug_graph_persistence.py`
  - `record_graph_timelapse.py`
  - `test_viz.py`

- [ ] **Update `.gitignore`**
  - Add: `*.log`
  - Add: `benchmark_viz_enhanced_memory/`
  - Add: `benchmark_viz_v5_memory/`
  - Add: `longmemeval_graph_memory/`
  - Add: `longmemeval_viz_enhanced_memory/`
  - Add: `longmemeval_memory_v5/`
  - Add: `dist/`

- [ ] **Remove stale print() statements**
  - `graph_store.py` line ~377: `print(f"Graph loaded: ...")`
  - `memory_store_v5.py` lines ~334-340: `print(f"[DEBUG] Extracting...")`
  - `memory_store_v5.py` line ~349: `print(f"\n[ERROR] Extraction failed: ...")`
  - `retrieval.py` (V4) lines 219-220: `print(f"Query entities: ...")` and `print(f"Related entities: ...")`

## Success Criteria

1. V5 LOCOMO F1 score > V4 LOCOMO F1 score (V4 baseline: 0.228-0.434)
2. V5 ingestion latency per turn < 2x V4 (no extra LLM calls for MemoryManager)
3. Zero `print()` debug statements in V5 code
4. All root-level `.log` files and debug scripts deleted
5. `sentence-transformers` and `cross-encoder` models work offline (no API key)

## Dependencies Added

```
sentence-transformers>=2.2.0  # Includes both SentenceTransformer and CrossEncoder
```

No new API keys or services required. Models download from HuggingFace on first use (~45MB total).
