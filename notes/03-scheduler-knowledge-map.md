# vLLM Scheduler 知识地图

## 第一层：vLLM Core Engine 架构鸟瞰

```
请求进来
  │
  ▼
API Server ──→ LLM Engine ──→ Scheduler ──→ Model Runner ──→ GPU Worker
                    ▲              │
                    │              ▼
                KvCacheManager  BlockAllocator(PagedAttention)
```

**Scheduler 的位置：** 它是 Engine 内部的"交通指挥"。每个 step 被调用一次，决定：
- 哪个 request 本轮能跑（分配 token budget）
- 哪个 request 要等等（留在 waiting）
- 哪个 request 要被挤出去（preempt）

> 先记住这个图就行，不用深究细节。这是你每次 dive 之前要看一眼的东西，提醒自己"我在哪"。

---

## 第二层：五个探索方向

每个方向都跟第一层的某个部分对应。从你最感兴趣的开始。

### 方向 A ── Scheduler 的资源调度逻辑（对应 Scheduler 核心）

- A1: scheduler 到底在调度什么？算力、显存还是 token？
- A2: token budget 是什么，怎么分给每个 request 的？
- A3: 为什么 v1 scheduler 说"没有 prefill 和 decode 的区分"？

### 方向 B ── 三个队列的流转（对应 Scheduler 输入输出）

- B1: running / waiting / preempted 三个队列的关系
- B2: 什么时候 waiting 的 request 会被跳过？
- B3: running → preempted 的触发条件

### 方向 C ── 一个 request 的完整旅程（对应整个流程）

- C1: add_request 进来 → 放哪？
- C2: schedule() 的每一步做了什么？
- C3: 什么时候从 waiting 变成 running？
- C4: 结束之后怎么清理？

### 方向 D ── Preempt 策略（对应 Scheduler 决策）

- D1: 什么时候需要 preempt？
- D2: preempt 选谁？（哪条策略）
- D3: preempt 之后 request 去哪？
- D4: v0 和 v1 scheduler 的 preempt 有什么区别？

### 方向 E ── PagedAttention + Scheduler（对应 Scheduler → BlockAllocator）

- E1: PagedAttention 解决什么问题？
- E2: scheduler 分配 token 时，kvcache block 怎么分配？
- E3: preempt 释放的 block 怎么回收？

---

## 检查标准

- [ ] 能不看代码，讲清第一层的架构图（2 分钟）
- [ ] 自己选的探索方向，能讲清 3 个以上的子问题（2 分钟）
- [ ] 每个子问题有一句话笔记在 notes/ 里

## 学习方法

1. 看一眼第一层，确认"我大概在哪"
2. 选一个当前最感兴趣的方向（不用按顺序）
3. 针对那个方向里的子问题，先猜答案，再看代码/问我
4. 写一句话笔记
5. 下次从另一个方向开始
