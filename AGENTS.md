# vllm-learning

Learning record / study project for [vLLM](https://github.com/vllm-project/vllm) — the high-throughput LLM serving engine (Python, C++, CUDA).

## Status

- **Just started.** Repo has a single initial commit (README, LICENSE, `.gitignore`).
- `third_party/` contains upstream repos as git submodules (shallow, `--depth 1`) for study.
- No custom Python/C++ code, no build config, no tests yet.
- The `.gitignore` was generated from a Rust template — expect it to be updated as Python artifacts (`__pycache__`, `*.pyc`, `venv/`, `.egg-info/`, `dist/`) are added.

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
│   └── vllm-ascend/    # https://github.com/vllm-project/vllm-ascend (submodule, --depth 1)
└── ...                 # your own notes / experiments go here
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

## Repository conventions

- No monorepo or multi-package structure — flat learning project.
- README is intentionally minimal (`# vllm-learning\nvllm learning record`).
- MIT licensed.

## For future agents

- This is not a vLLM fork or production deployment — it's a **personal study sandbox**.
- The user is Edge51 (liuyuanyuanedge@163.com) — working from `/home/Edge51/Projects/vllm-learning`.
- Do NOT modify files inside `third_party/` — those are upstream submodules for reference only.
- Changes to `.gitignore` are expected as the project accumulates Python artifacts.
- If learning experiments need vLLM source changes, copy the relevant files out of `third_party/` first.
