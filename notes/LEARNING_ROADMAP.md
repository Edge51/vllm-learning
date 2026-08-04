# vLLM 学习地图

目标：系统掌握 vLLM 推理引擎的架构与核心设计，能胜任 vLLM 相关岗位的工作。

你目前在 ascend-vLLM 组维护 function call 特性。这份地图从你的实际工作出发，逐层深入。

---

## 如何使用

```
每个阶段的结构：
├── 学习目标      → 学完后你应该能做什么
├── 核心问题      → 用问题引导你思考（苏格拉底式）
├── 学习内容      → 按顺序看什么、读什么
├── 代码漫游路径  → 在 third_party/ 里读哪些文件
└── 🗣 费曼检查点  → 用自己的话讲一遍，能讲出来才算会
```

**标记说明：**
- ⭐ = 必须掌握（面试必问、工作必用）
- 🔧 = 你工作直接相关的（function call / ascend 相关）
- 🧪 = 实验建议（动手做一下）

---

## 📍 当前位置

```
P0 [████████░░] ✅  请求处理流程、引擎主循环 step()
P1 [████████░░] ✅  Prefill/Decode 区分、KV cache 概念
P2 [████████░░] ✅  PagedAttention、block table、block 生命周期
P3 [████████░░] ✅  连续批处理与调度细节
P4 [█████████░] 🔧  function call（主干+核心算法+共享状态已覆盖）
P5 [███████░░░] 🔧  分布式推理（全部已读，**待复述验证掌握度**）
P6 [░░░░░░░░░░] ❌  Ascend 适配与 CANN
P7 [░░░░░░░░░░] ❌  面试准备与实战
```

下一站：**Phase 5 收尾（通信原语细读 + 配置案例）→ Phase 6 Ascend，或从面试清单复述开始**

---

## Phase 0：先搞懂你在维护什么 ✅ 已完成

### 学习目标

画出 vLLM 处理一次请求的完整流程图，从 HTTP 请求进来到响应出去。

### 已覆盖内容

- API Server（entrypoints/openai/api_server.py）接收请求
- Chat 请求处理（serving_chat.py → jinja 模板渲染）
- Tokenize → Scheduler → EngineCore.step() 主循环
- ModelRunner 执行 → 采样 → Detokenize → 响应

### 代码漫游路径（v1，实际文件路径）

```
third_party/vllm/vllm/v1/engine/core.py        # EngineCore.step() 主循环
third_party/vllm/vllm/v1/engine/core.py        # EngineCoreOutput 输出处理
third_party/vllm/vllm/v1/serial_utils.py        # 序列化/反序列化
```

### 相关笔记

- `notes/00-request-flow.md` — 请求处理流程图与代码对照

---

## Phase 1：LLM 推理基础 ✅ 已完成

### 学习目标

- 理解 Transformer 的解码过程：prefill 和 decode 两个阶段
- 理解什么是 KV cache，为什么需要它
- 理解 self-attention 的计算过程（Q、K、V）

### 核心问题

1. 模型生成第一个 token 和后续 token 的计算过程有什么不同？
2. 为什么不能每生成一个 token 就重算一遍前面的注意力？
3. KV cache 到底 cache 了什么？cache 在哪？
4. Prefill 阶段和 decode 阶段的计算量差异在哪？

### 学习内容

| 顺序 | 主题 | 建议资源 |
|---|---|---|
| 1 | Transformer 架构概览 | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) |
| 2 | LLM 推理过程 | [LLM 推理入门](https://huggingface.co/blog/llm-inference) |
| 3 | Self-attention 和 KV cache | 读完任何一篇讲 KV cache 的文章 |
| 4 | 为什么需要 PagedAttention | [vLLM 官方论文](https://arxiv.org/abs/2309.06180) 前 2 节 |

### 代码漫游路径

```
# 看代码前先理解：vLLM 怎么组织一次推理
third_party/vllm/vllm/worker/model_runner.py  # 模型执行器
  └─ execute_model()                            # 每次 forward 的入口
  └─ prepare_inputs()                          # prefill/decode 的不同准备

third_party/vllm/vllm/attention/             # attention 后端
  └─ attention.py                              # attention 接口
```

### 🧪 动手实验

在 `examples/` 下写一个脚本：`pip install vllm` 后，用 vLLM 跑一次离线推理，打印出每步生成的 token 和 logits。

### 相关笔记

- `notes/01-prefill-decode-basics.md` — Prefill/Decode 对比与 KV Cache 解释

### 🗣 费曼检查点

> "用最通俗的话讲，Transformer 是怎么一个字一个字写出回答的？prefill 和 decode 有什么区别？KV cache 存了什么、为什么能加速？"

---

## Phase 2：vLLM 核心架构 — PagedAttention 与显存管理 ✅ 已完成

### 学习目标

- 理解 PagedAttention 的核心思想：操作系统的分页思想用在 KV cache 上
- 理解逻辑块和物理块通过 block table 映射
- 理解 Block 的完整生命周期：分配 → 使用 → 释放

### 已覆盖内容

- **逻辑 block vs 物理 block**：逻辑连续性由 list 下标保证，物理 block_id 可以不连续
- **Block table**：`single_type_kv_cache_manager.py:73` `req_to_blocks: dict[str, list[KVCacheBlock]]`
- **分配链路**：`scheduler.allocate_slots()` → `KVCacheManager.allocate_slots()` → `coordinator.allocate_new_blocks()` → `SingleTypeKVCacheManager.allocate_new_blocks()` → `block_pool.get_new_blocks()` → `FreeKVCacheBlockQueue.popleft_n()`
- **释放链路**：`scheduler._preempt_request()` → `kv_cache_manager.free()` → `SingleTypeKVCacheManager.free()` → `block_pool.free_blocks()` → `FreeKVCacheBlockQueue.append_n()`
- **ref_cnt**：分配时 +1，释放时 -1，ref_cnt == 0 才回 free list
- **KVCacheBlock**（`kv_cache_utils.py:114`）：block_id、ref_cnt、双向链表指针
- **prefix cache 共享**：多个 request 可共享同一 block，ref_cnt > 1
- **1024 token 示例**：16 block_size → 64 blocks，preempt 时全部回 free list

### 关键代码路径

```
third_party/vllm/vllm/v1/core/block_pool.py
  └─ BlockPool (line 130)
  └─ get_new_blocks() (line 322)
  └─ free_blocks() (line 408)
third_party/vllm/vllm/v1/core/single_type_kv_cache_manager.py
  └─ SingleTypeKVCacheManager (line 30)
  └─ allocate_new_blocks() (line 242)
  └─ free() (line 303)
third_party/vllm/vllm/v1/core/kv_cache_utils.py
  └─ KVCacheBlock (line 114)
  └─ FreeKVCacheBlockQueue (line 162)
third_party/vllm/vllm/utils/math_utils.py
  └─ cdiv() (line 10)
```

### 相关笔记

- `notes/02-block-lifecycle.md` — Block 生命周期完整追踪

### 🗣 费曼检查点

> "PagedAttention 是什么？逻辑 block 和物理 block 怎么映射？一个 1024 token 的请求 prefill 时分配了多少 block？preempt 时 block 去了哪里？"

---

## Phase 3：调度与批处理 ✅ 已完成

### 学习目标

- 理解 Continuous Batching 的核心机制
- 理解 vLLM v1 调度器的三个队列（Running / Resumed / Waiting）和 token budget 分配
- 理解 Chunked Prefill 的原理和配置

### 已覆盖内容

- **Scheduler 哲学**：没有 prefill/decode 分支，只有 `num_new_tokens` 的差值计算
- **Running 队列调度**（`scheduler.py:387-522`）：遍历 running，算 `num_new_tokens`，分配 KV cache，不够踢人
- **Waiting 队列调度**（`scheduler.py:567-840`）：新请求首次分配，prefix cache 匹配，`long_prefill_token_threshold` 截断
- **Preemption**（`scheduler.py:965-985`）：踢人 → `kv_cache_manager.free()` → 放回 waiting
- **Token Budget**：三个阶段共享同一个 `token_budget`，Running 优先
- **Chunked Prefill**：`long_prefill_token_threshold` 限制单个 request 每步 prefill 上限，`enable_chunked_prefill` 控制是否允许分批
- **Block 生命周期**：`block_pool.get_new_blocks()` → `FreeKVCacheBlockQueue.popleft_n()`；释放走 `free_blocks()` → `append_n()`，ref_cnt 控制是否回 pool
- **配置项**（`config/scheduler.py`）：`max_num_batched_tokens`、`max_num_seqs`、`long_prefill_token_threshold`、`enable_chunked_prefill`
- **执行链路**（`gpu_model_runner.py:3787-4116`）：_update_states → _prepare_inputs → _preprocess → _model_forward → compute_logits

### 关键代码路径

```
# v1 调度器（vllm v0.20.2）
third_party/vllm/vllm/v1/core/sched/scheduler.py
  └─ schedule() (line 352)                     # 主入口
  └─ _preempt_request() (line 965)             # 踢人
  └─ _free_request() (line 1826)               # 完成释放

# KV cache 管理
third_party/vllm/vllm/v1/core/block_pool.py
  └─ BlockPool.get_new_blocks() (line 322)     # 分配
  └─ BlockPool.free_blocks() (line 408)        # 释放
third_party/vllm/vllm/v1/core/single_type_kv_cache_manager.py
  └─ allocate_new_blocks() (line 242)          # 计算 + 调 get_new_blocks
  └─ free() (line 303)                         # pop + 调 free_blocks
third_party/vllm/vllm/v1/core/kv_cache_utils.py
  └─ FreeKVCacheBlockQueue (line 162)          # 空闲块双向链表

# 配置
third_party/vllm/vllm/config/scheduler.py
  └─ SchedulerConfig (line 26)

# 模型执行
third_party/vllm/vllm/v1/worker/gpu_model_runner.py
  └─ execute_model() (line 3787)
```

### 相关笔记

- `notes/02-block-lifecycle.md` — 1024 seq_len 请求的 block 完整生命周期（P2，与调度相关）
- `notes/03-scheduler-knowledge-map.md` — Scheduler 架构知识地图

### 🗣 费曼检查点

> "连续批处理解决了什么问题？调度器里没有 if prefill / if decode 分支，那 prefill 和 decode 的本质区别是什么？token_budget 是怎么在多个 request 之间分配的？"

---

## Phase 4：深入 function call 🔧 主链路 + 核心算法已完成

### 学习目标（已达成）

- ✅ 从源码层面理解 function call 的完整实现
- ✅ 理解 jinja 模板的作用和渲染时机
- ✅ 理解 tool_parser 的实现（包括 JSON Schema 和 XML 两种路径）
- ✅ 能定位和修复 function call 相关的问题
- ✅ 理解投机解码对 tool parser 的影响
- ✅ 理解 `extract_tool_call_required_streaming` 算法 + `partial_json_loads` 配合
- ✅ 理解 `_parse_tool_calls_from_content` 非流式 5 分支决策树
- ✅ 理解 `_filter_delta_text` bracket 过滤机制
- ✅ 理解 `make_tool_call_id` 的两种格式和 `finish_reason` 分支逻辑
- ✅ 理解 `prev_tool_call_arr` / `streamed_args_for_tool` 共享状态设计

### 已覆盖内容

详细内容见 `notes/04-function-call.md`（§1-13 全链路分析）和 `notes/04-streaming-branches.md`（流式分支架构）。

核心链路：

```
create_chat_completion()
  └→ render_chat_request()    → jinja 渲染 messages+tools → prompt
      └→ preprocess_chat()
          ├→ reasoning_parser.adjust_request()
          └→ tool_parser.adjust_request()   → 设置 structured_outputs / skip_special_tokens
  └→ engine_client.generate() → AsyncLLM → EngineCore
  └→ (stream) chat_completion_stream_generator()
      ├→ tool_parser.extract_tool_calls_streaming()  ← XML parser 逐 token 解析
      └→ extract_tool_call_required_streaming()      ← required/named JSON 路径
  └→ (non-stream) chat_completion_full_generator()
      └→ _parse_tool_calls_from_content()            ← 5 分支决策树
```

### 关键发现

| 主题 | 要点 |
|---|---|
| **Jinja 渲染** | vLLM 侧只加载模板字符串；真正编译执行在 HuggingFace `apply_chat_template()` |
| **adjust_request** | 是 **pre-processing**（改请求参数），不是后处理 |
| **JSON Schema** | `tool_choice="required"` 时构建，约束模型输出 JSON 格式 |
| **投机解码影响** | `delta_text` 批量化 → XML 状态机容易不同步 |
| **无状态解析** | JSON 格式（`partial_json_loads`）天然不受投机影响 |
| **技术债** | Mistral / Harmony 的 `if/else` 散落主线，可读性差 |
| **_filter_delta_text** | bracket level 扫描，过滤末尾不完整 JSON fragment，解决 spec-decode 截断问题 |
| **non-streaming 5 分支** | `_parse_tool_calls_from_content` 按 tool_choice + 是否有 parser + finish_reason 决策 |
| **make_tool_call_id** | 两种格式：`call_<uuid>` 或 `functions.<name>:<idx>` |
| **finish_reason 转换** | named tool_choice 时 `"tool_calls"` → `"stop"`（对客户端透明） |

### 影响 function call 的特性清单

| 特性 | 等级 | 说明 |
|---|---|---|
| Speculative decoding | 🔴 高 | delta_text 批量化，XML parser 状态机不同步 |
| 并行 tool call | 🔴 高 | 多 tool call 边界管理 |
| Reasoning + tool 混合 | 🔴 高 | 两个 parser 共享 delta_message 流 |
| Grammar/Guided decoding | 🔴 高 | 约束格式必须和 parser 一致 |
| Chunked prefill | 🟡 中 | 空 delta 处理 |
| n > 1 | 🟡 中 | 多 choice 独立状态 |
| Multi-modal | 🟡 中 | 多模态 token 穿插 |
| EOS 处理 | 🟡 中 | 截断处理 |

### 相关笔记

- `notes/04-function-call.md` — 完整 Phase 4 学习文档（§1-14）
- `notes/04-streaming-branches.md` — 流式分支架构（3 分类 + 决策树）

---

## Phase 5：分布式推理（3-5 天）🔄 进行中

### 学习目标

- 理解为什么需要分布式推理（模型太大放不进一张卡）
- 理解 tensor parallelism 和 pipeline parallelism 的区别
- 理解 vLLM 的分布式架构

### 核心问题

1. 一个 70B 模型大概占多少显存？一张 A100 80GB 能放下吗？ — ✅ 已答（140GB，放不下）
2. Tensor parallelism 是怎么把一个 transformer 层切到多张卡上的？ — ✅ 已答（列切/行切 + all-reduce）
3. Pipeline parallelism 又是什么？它和 tensor parallelism 有什么区别？ — ✅ 已答（PP 分层传激活，TP 切参数）
4. vLLM 用哪种并行方式？什么时候需要？ — ✅ 已答（TP×PP×EP 组合，看模型/显存）
5. 多卡通信是怎么做的？（NCCL？CANN 对应的通信库是什么？） — ⚠️ 部分（collective_rpc/MQ 已读，NCCL/CANN 细节待补）
6. 🔧 多卡执行流：EngineCore.step() 怎么把调度结果派发给所有 worker？ — ✅ 已答（collective_rpc → MQ）
7. 🔧 Disaggregated Prefill/Decode 和 PP 有什么区别？Mooncake 怎么传 KV cache？ — ✅ 已答（RDMA 传完整 KV）

### 学习内容

| 顺序 | 主题 | 状态 |
|---|---|---|
| 1 | 模型并行 vs 数据并行 vs 流水线并行 | ✅ |
| 2 | Tensor parallelism（Megatron-LM 的分片方式） | ✅ |
| 3 | vLLM 的分布式实现（组拓扑） | ✅ |
| 4 | Pipeline Parallel + AsyncIntermediateTensors | ✅ |
| 5 | MoE / Expert Parallel（SparseMoeBlock 入口） | ✅ |
| 6 | AsyncLLM 多卡执行流（collective_rpc/MQ/Worker） | ✅ |
| 7 | Disaggregated Prefill/Decode（Mooncake） | ✅ |
| 8 | 通信原语细读（communication_op.py） | ❌ 待学 |
| 9 | 多卡配置案例（实际部署怎么选 TP/PP/EP） | ❌ 待学 |

### 代码漫游路径

```
third_party/vllm/vllm/distributed/          # 分布式基础设施
  └─ parallel_state.py                       # 并行状态管理（5D 组拓扑、isend/irecv）
  └─ comms.py / communication_op.py          # 通信接口（部分）
  └─ elastic_ep/                             # 弹性 Expert Parallel（待学）

third_party/vllm/vllm/v1/executor/
  └─ multiproc_executor.py                   # collective_rpc → MQ 派发（✅ 已读:306/339）

third_party/vllm/vllm/v1/worker/
  └─ gpu_worker.py                           # GPUWorker.execute_model（✅ 已读:753）
  └─ kv_connector_model_runner_mixin.py      # Mooncake 拉取 KV（✅ 已读:102）

third_party/vllm/vllm/distributed/kv_transfer/
  └─ kv_connector/v1/mooncake/mooncake_connector.py   # MooncakeConnector（✅ 已读:461/703/978）
  └─ kv_connector/v1/base.py                 # KVConnectorBase_V1 抽象

third_party/vllm/vllm/model_executor/layers/
  └─ fusions/fused_moe/                      # FusedMoE kernel 层（待学）
```

### 🗣 费曼检查点

> "如果模型太大放不进一张 GPU，怎么用多张卡来跑推理？Tensor parallelism 和 pipeline parallelism 各自怎么切分计算？"
> → 已学待复述（见 07-interview-prep.md 题 9-16）

---

## Phase 6：Ascend 适配与 CANN 层（3-5 天）🔧

### 学习目标

- 理解 ascend-vLLM 作为硬件插件的架构
- 理解 CANN 和 CUDA 的对应关系
- 能看懂 ascend-vLLM 的适配代码

### 核心问题

1. vLLM 的硬件插拔接口（hardware pluggable RFC）是什么？
2. ascend-vLLM 替换了 vLLM 的哪些组件？
3. CANN 的算子调用和 CUDA kernel 调用有什么对应关系？
4. ascend-vLLM 的性能瓶颈一般在哪里？
5. ModelArts-Lab 的 patch 和 ascend-vLLM 的关系是什么？

### 代码漫游路径

```
third_party/vllm-ascend/
  └── vllm/worker/                           # ascend 版 worker
  └── vllm/attention/                        # ascend 版 attention
  └── vllm/model_executor/                   # 模型执行
  └── csrc/                                  # ascend C++/CANN kernel
  └── requirements.txt                       # 依赖对比

third_party/vllm-cloud-main/
  └── (ModelArts-Lab 对 vllm-ascend 的 patch)
```

### 🗣 费曼检查点

> "vLLM 在 GPU 上跑和 ascend-vLLM 在 NPU 上跑，从架构角度看有什么区别？CANN 在其中的角色是什么？"

---

## Phase 7：面试准备与实战（贯穿全程）

### 常见面试题分类

| 类别 | 典型问题 |
|---|---|
| **推理基础** | 讲一下 Transformer 的解码过程；KV cache 怎么实现的；prefill 和 decode 的区别 |
| **vLLM 核心** | PagedAttention 解决了什么问题；连续批处理怎么工作的；调度器如何调度 |
| **系统设计** | 设计一个 LLM 推理服务；如何优化吞吐/延迟；怎么处理长序列 |
| **分布式** | 模型太大放不进一张卡怎么办；tensor parallelism 的原理 |
| **部署** | 怎么做 vLLM 的性能调优；max_num_seqs 怎么调；显存不够怎么办 |

### 推荐项目（在 experiments/ 下做）

1. **从零实现 PagedAttention 的逻辑** — 不用 CUDA，用 Python 模拟 block table 和物理块分配
2. **写一个最小调度器** — 模拟 prefill/decode 调度决策
3. **分析一次推理的显存分布** — 用 `nvidia-smi` 或 CANN 工具观察
4. **对比 vLLM 和 ascend-vLLM 的代码差异** — 找出所有适配点

---

## 学习节奏建议

```
每周目标（快速推进版）：
  周一-周二：Phase 1 + 代码漫游
  周三-周四：Phase 2 + 写笔记
  周五：Phase 3 + 🗣 费曼复述本周内容
  
两周一次：
  🧪 动手实验
  📝 写一篇笔记文章放在 notes/
```

---

## 代码阅读技巧

```
1. 从入口开始：永远先找 api_server.py 或 LLM 类
2. 只看主线：第一次读跳过错误处理和边界条件
3. 打印大法：加 print 看变量（vLLM 是 Python，这是最大优势）
4. 对比阅读：同时打开 GPU 版和 ascend 版对比差异
5. 带着问题读：不要漫无目的地读代码，先问自己一个问题
```

---

## 最后几句话

- **你现在的工作就是最好的学习素材** — function call 的每个 bug 都是一个深入了解 vLLM 的机会
- **不要试图一次看懂所有代码** — vLLM 代码量很大，先看主线流程
- **费曼法是最有效的检验** — 如果你不能简单讲清楚，说明还没真懂
- **每张图胜过千言** — 每个 phase 学完画一张图

在 `notes/` 下建子目录写笔记，在 `experiments/` 下动手改代码。你不是在读一个项目，你是在**解剖**它。
