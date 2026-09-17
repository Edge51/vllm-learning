# vllm-learning

Learning record / study project for [vLLM](https://github.com/vllm-project/vllm).

## User language

- Communicate with the user in Chinese (zh-CN). Code, file paths, and identifiers stay in English.

## Layout

```
third_party/   # upstream submodules for study (vllm, vllm-ascend, vllm-cloud-main)
notes/         # study notes (LEARNING_ROADMAP.md = live progress)
examples/      # example scripts using vLLM API
experiments/   # deeper experiments
```

## Hard rules

- **Never modify `third_party/`** — copy files out before experimenting.
- Learning counts only when the user **personally reads source code and can explain it**. Assistant-driven explanations don't count until verified by reading.
- One small code path at a time.

## Environment

- miniconda env `vllm` (Python 3.12) — host Python 3.14 incompatible with vLLM.
- GPU: RTX 4060 Ti 8GB.

## Learning progress

- Live progress: `notes/LEARNING_ROADMAP.md`
- Notes index: `notes/*.md`
- Interview prep: `notes/07-interview-prep.md`

## Learning method

- Use the `learn` skill to guide study sessions (Socratic questioning + Feynman technique + dual-track: mainline/curiosity).
- Iron rule: nothing counts as learned until the user personally reads source code AND restates it in their own words.
