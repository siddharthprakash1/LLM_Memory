# LLM Memory Benchmark Suite

A comprehensive benchmark suite for evaluating hierarchical long-term memory systems, based on **industry-standard** evaluation protocols.

## 🎯 What We Measure

Based on **MemoryAgentBench**, **Mem0**, **TiMem**, and **GoodAI LTM** benchmarks:

### Four Core Competencies

1. **Accurate Retrieval** - Can it find the right memory?
2. **Test-time Learning** - Can it integrate new information during interaction?
3. **Long-range Understanding** - Does it handle temporal relations and multi-hop reasoning?
4. **Conflict Resolution** - How well does it manage contradictory/outdated memories?

### Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match** | Perfect string match accuracy |
| **Contains Match** | Answer contained in retrieval |
| **F1 Score** | Token-level F1 (standard for QA) |
| **P95 Latency** | 95th percentile response time |
| **Compression Ratio** | History tokens / Memory tokens |
| **Forgetting Slope** | How fast memories degrade |
| **Composite Score** | Weighted aggregate (0-100) |

## 🚀 Quick Start

```bash
# Quick benchmark (for development)
python -m benchmarks.benchmark_memory --quick

# Full benchmark (publication-grade)
python -m benchmarks.benchmark_memory --full

# Without Ollama (baseline only)
python -m benchmarks.benchmark_memory --no-memory-system

# List available scenarios
python -m benchmarks.benchmark_memory --list-scenarios
```

## 📊 Benchmark Scenarios

| Scenario | Description | Difficulty |
|----------|-------------|------------|
| `single_hop` | Direct fact retrieval | Easy |
| `multi_hop` | Inference across multiple memories | Hard |
| `temporal` | Time-based queries and updates | Medium |
| `conflict` | Contradictory information handling | Hard |
| `preference` | User preference consistency | Medium |
| `episodic` | Specific event recall | Medium |
| `long_range` | Extended multi-session coherence | Hard |

## 🔧 Configuration

```bash
# Custom scenarios
python -m benchmarks.benchmark_memory --scenarios single_hop,temporal,conflict

# More samples (better statistical significance)
python -m benchmarks.benchmark_memory --samples 100 --runs 5

# Different model
python -m benchmarks.benchmark_memory --model qwen2.5:32b
```

## 📈 Comparing with Baselines

Compare your system against published baselines:

```bash
# Compare with Mem0 reported numbers
python -m benchmarks.benchmark_memory --compare mem0_reported

# Compare with TiMem reported numbers
python -m benchmarks.benchmark_memory --compare timem_reported

# Compare with simple RAG baseline
python -m benchmarks.benchmark_memory --compare simple_rag

# Also run local baseline for fair comparison
python -m benchmarks.benchmark_memory --baseline
```

### Available Baselines

| Baseline | Description |
|----------|-------------|
| `full_history` | No compression, full context (upper bound) |
| `simple_rag` | Basic vector search RAG |
| `mem0_reported` | Mem0 paper reported numbers |
| `timem_reported` | TiMem paper reported numbers |

## 📝 Output

Benchmarks generate:

1. **JSON Results** (`reports/LLM-Memory_TIMESTAMP.json`)
   - Full metrics breakdown
   - Per-sample results
   - Statistical aggregations

2. **Markdown Report** (`reports/LLM-Memory_TIMESTAMP.md`)
   - Executive summary
   - Performance tables
   - Methodology notes

3. **Console Summary**
   - ASCII bar charts
   - Overall rating (★★★★★)

## 🏆 Scoring

The **Composite Score** (0-100) is calculated as:

| Component | Weight |
|-----------|--------|
| Retrieval Accuracy | 30% |
| Latency Score (inverse p95) | 20% |
| Compression Efficiency | 15% |
| Forgetting Resistance | 15% |
| Multi-hop Accuracy | 20% |

### Rating Scale

| Score | Rating |
|-------|--------|
| 80+ | ★★★★★ EXCELLENT |
| 60-79 | ★★★★☆ GOOD |
| 40-59 | ★★★☆☆ AVERAGE |
| 20-39 | ★★☆☆☆ BELOW AVERAGE |
| <20 | ★☆☆☆☆ NEEDS IMPROVEMENT |

## 🔬 For Research

For publication-grade benchmarks:

```bash
python -m benchmarks.benchmark_memory \
  --full \
  --runs 5 \
  --samples 100 \
  --compare mem0_reported \
  --baseline \
  --output benchmarks/reports/publication
```

This ensures:
- All 7 scenarios tested
- 100 samples per scenario
- 5 runs for statistical significance
- Comparison with published baselines
- Local baseline for fair comparison

## 📚 References

- **MemoryAgentBench**: [arXiv:2507.05257](https://arxiv.org/abs/2507.05257)
- **Mem0 Benchmark**: [arXiv:2504.19413](https://arxiv.org/abs/2504.19413)
- **TiMem**: [arXiv:2601.02845](https://arxiv.org/abs/2601.02845)
- **GoodAI LTM**: [goodai.com/ltm-benchmark](https://www.goodai.com/introducing-goodai-ltm-benchmark/)
- **LoCoMo**: Long-Context Memory Evaluation
