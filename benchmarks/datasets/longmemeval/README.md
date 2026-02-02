# LongMemEval Dataset

This directory contains the LongMemEval benchmark dataset.

## Download Instructions

The dataset file is too large for GitHub (264MB). Download it separately:

### Option 1: Direct Download from HuggingFace

```bash
# Using curl
curl -L https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json \
  -o benchmarks/datasets/longmemeval/longmemeval_s_cleaned.json

# Or using wget
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json \
  -O benchmarks/datasets/longmemeval/longmemeval_s_cleaned.json
```

### Option 2: Using Python

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="xiaowu0162/longmemeval-cleaned",
    filename="longmemeval_s_cleaned.json",
    repo_type="dataset",
    local_dir="benchmarks/datasets/longmemeval"
)
```

## Dataset Details

- **File**: `longmemeval_s_cleaned.json`
- **Size**: 264 MB
- **Questions**: 500
- **Average tokens per question**: ~115k
- **Average sessions per question**: 30-40

## Citation

```bibtex
@article{wu2024longmemeval,
  title={LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory}, 
  author={Di Wu and Hongwei Wang and Wenhao Yu and Yuwei Zhang and Kai-Wei Chang and Dong Yu},
  year={2024},
  eprint={2410.10813},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
