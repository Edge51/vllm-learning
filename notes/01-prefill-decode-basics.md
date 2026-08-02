# Phase 1: LLM 推理基础 — Prefill / Decode 与 KV Cache

## Prefill vs Decode

| | Prefill | Decode |
|---|---|---|
| 何时发生 | 请求第一次进入 | 已有 KV cache 后 |
| 计算量 | 大量（并行算所有 token 的 KV） | 小（只算 1 个新 token） |
| GPU 用途 | 矩阵乘法（计算密集型） | 显存读取（带宽密集型） |
| `num_scheduled_tokens` | > 1（通常几百） | = 1 |

## KV Cache

- 缓存的是每层 attention 的 **K 和 V**（不是 Q）
- 一旦算好，后续 decode 直接读取，不用重算
- Q 每轮都变，不能缓存

## 核心问题

1. 模型生成第一个 token 和后续 token 的计算过程有什么不同？
2. 为什么不能每生成一个 token 就重算一遍前面的注意力？
3. KV cache 到底 cache 了什么？cache 在哪？
4. Prefill 阶段和 decode 阶段的计算量差异在哪？

## 🗣 费曼检查点

> "用最通俗的话讲，Transformer 是怎么一个字一个字写出回答的？prefill 和 decode 有什么区别？KV cache 存了什么、为什么能加速？"
