# vllm-learning

Learning record / study project for [vLLM](https://github.com/vllm-project/vllm) — the high-throughput LLM serving engine (Python, C++, CUDA).

## What this repo is for

- Studying vLLM internals: architecture, batching, KV cache management, tensor parallelism, PagedAttention, continuous batching.
- Experimenting with custom models, kernels, or serving configurations.
- Reproducing vLLM features from scratch as a learning exercise.

## Directory layout

```
vllm-learning/
├── AGENTS.md           # this file
├── .gitmodules         # submodule index
├── third_party/        # upstream repo submodules for study
│   ├── vllm/           # https://github.com/vllm-project/vllm (submodule, --depth 1)
│   ├── vllm-ascend/    # https://github.com/vllm-project/vllm-ascend (submodule, --depth 1)
│   └── vllm-cloud-main/ # https://github.com/huaweicloud/ModelArts-Lab branch:vllm/vllm-cloud-main (submodule, --depth 1)
├── notes/              # study notes, architecture docs (LEARNING_ROADMAP.md = live progress)
├── examples/           # your own example scripts using vLLM API
├── experiments/        # deeper experiments, custom modifications
└── ...                 # more as needed
```

## Commands

```bash
# install vllm
pip install vllm

# after cloning vllm-learning, pull submodules
git submodule update --init --depth 1

# develop from upstream source
pip install -e third_party/vllm
pip install -e third_party/vllm-ascend
```

## Hard rules

- **Do NOT modify files inside `third_party/`** — upstream submodules for reference only. If an experiment needs vLLM source changes, copy the files out first.
- This is a personal study sandbox, not a vLLM fork or production deployment.
- Learning counts only when the user **personally reads source code** and can explain it. Assistant-driven explanations don't count until verified by reading.
- Prefer: one small code path at a time, quick explanation of blockers, immediate return to code.

## Learning progress

- **Live progress** (phases, next step): `notes/LEARNING_ROADMAP.md`
- **Notes index**: `notes/*.md` — one file per topic (request flow, scheduler, function call, distributed inference, etc.)
- **Interview prep** (Phase 7): `notes/07-interview-prep.md`

## Environment

- Python 3.14 on host — incompatible with vLLM (needs 3.10-3.12)
- Workaround: miniconda env `vllm` with Python 3.12
- GPU: RTX 4060 Ti 8GB
