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
P0 [████████░░] 基本完成  请求处理流程、引擎主循环 step()
P1 [██████░░░░] 进行中    Prefill/Decode 区分、KV cache
P2 [░░░░░░░░░░] 未开始    PagedAttention 与显存管理
P3 [░░░░░░░░░░] 未开始    连续批处理与调度细节
P4 [░░░░░░░░░░] 未开始    深入 function call
P5 [░░░░░░░░░░] 未开始    分布式推理
P6 [░░░░░░░░░░] 未开始    Ascend 适配与 CANN
P7 [░░░░░░░░░░] 未开始    面试准备与实战
```

下一站：**Phase 2 — PagedAttention**

---

## Phase 0：先搞懂你在维护什么（1-2 天）✅ 基本完成

不用急着补理论，先从你每天都在接触的东西开始。

### 学习目标

能画出 vLLM 处理一次请求的**完整流程图**，从 HTTP 请求进来到响应出去。

### 核心问题

带着这些问题去读代码：

1. 用户发来的请求体长什么样？vLLM 在哪里接收的？
2. function call 的 tools 信息是怎么塞进 prompt 的？（jinja 渲染）
3. 模型输出的文本怎么被识别为 function call？（tool_parser）
4. 最终响应是怎么组装回去的？

### 代码漫游路径

```
third_party/vllm/vllm/entrypoints/       # API 入口
third_party/vllm/vllm/entrypoints/openai/  # OpenAI 兼容 API
  └─ api_server.py                        # HTTP 服务入口
  └─ serving_chat.py                      # chat 请求处理
third_party/vllm/vllm/entrypoints/llm.py  # LLM 类（离线推理入口）
third_party/vllm/vllm/transformers/       # tokenizer、detokenizer
```

function call 相关（大概率在 ascend-vLLM 或 ModelArts-Lab patch 里）：

```
第三_party/vllm-ascend/                    # ascend 适配层
third_party/vllm-cloud-main/              # 华为云 patch
  └─ 搜索 jinja / tool_parser / function_call
```

### 🗣 费曼检查点

> "当我用 OpenAI SDK 调一个 chat completion 请求时，vLLM 从收到请求到返回，中间经历了哪些步骤？画出流程图。"

---

## Phase 1：LLM 推理基础（3-5 天）🔄 进行中

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

### 🗣 费曼检查点

> "用最通俗的话讲，Transformer 是怎么一个字一个字写出回答的？prefill 和 decode 有什么区别？KV cache 存了什么、为什么能加速？"

---

## Phase 2：vLLM 核心架构 — PagedAttention 与显存管理（3-5 天）

### 学习目标

- 理解 PagedAttention 的核心思想：操作系统的分页思想用在 KV cache 上
- 理解 Block Manager 怎么管理显存
- 理解物理块和逻辑块的概念

### 核心问题

1. 传统 KV cache 有什么问题？（显存碎片、预分配浪费）
2. PagedAttention 怎么解决这些问题？
3. 一个请求的 KV cache 是怎么从逻辑块映射到物理块的？
4. Block table 是什么？什么时候更新？
5. Copy-on-write 在什么场景下发生？

### 学习内容

| 顺序 | 主题 | 建议资源 |
|---|---|---|
| 1 | PagedAttention 论文 → 重点读 3-4 节 | [vLLM 论文](https://arxiv.org/abs/2309.06180) |
| 2 | 官方 blog 文章 | [vLLM: PagedAttention 介绍](https://blog.vllm.ai/2023/06/20/vllm.html) |
| 3 | 显存管理概览 | vLLM 文档的显存部分 |

### 代码漫游路径

```
third_party/vllm/vllm/core/                 # vLLM 核心逻辑
  └─ block_manager.py                        # 块管理器 ⭐
  └─ block_table.py                          # 块表管理
  └─ block.py                                # 块的数据结构

third_party/vllm/vllm/worker/
  └─ cache_engine.py                         # KV cache 分配

third_party/vllm/csrc/attention/            # CUDA kernel（先看接口）
  └─ attention_kernel.cuh                    # PagedAttention kernel
```

### 🧪 动手实验

计算一下：使用 Llama-7B（层数 32，hidden_size 4096，FP16），batch_size=4，max_seq_len=4096，KV cache 需要多少显存？

### 🗣 费曼检查点

> "PagedAttention 是什么？它和操作系统的虚拟内存有什么相似之处？Block table 是怎么起作用的？"

---

## Phase 3：调度与批处理（3-5 天）

### 学习目标

- 理解 Continuous Batching（持续批处理）是什么、为什么重要
- 理解 vLLM 调度器的工作原理
- 理解 prefill 和 decode 怎么被调度

### 核心问题

1. 没有连续批处理的时候，vLLM 是怎么做推理的？
2. 连续批处理允许"中途插队"——新的请求进来时，正在跑的 batch 怎么办？
3. 调度器怎么决定当前这步应该跑 prefill 还是 decode？
4. 什么是 chunked prefill？为什么需要它？
5. 什么是 max_num_seqs、max_model_len？它们怎么影响调度？

### 学习内容

| 顺序 | 主题 |
|---|---|
| 1 | 连续批处理概念（先看这篇经典博客） |
| 2 | vLLM 调度器设计 |
| 3 | Chunked prefill 的引入原因 |

### 代码漫游路径

```
third_party/vllm/vllm/core/scheduler.py     # 调度器 ⭐
  └─ schedule()                              # 主调度逻辑
  └─ _schedule_prefills()                    # prefill 调度
  └─ _schedule_decodes()                     # decode 调度

third_party/vllm/vllm/v1/                   # v1 调度器（更新版本）
  └─ engine/
    └─ scheduler.py

third_party/vllm/vllm/config.py
  └─ SchedulerConfig                         # 调度器配置
```

### 🗣 费曼检查点

> "连续批处理解决了什么问题？调度器在每次 step 时怎么决定跑哪个请求的哪个阶段？"

---

## Phase 4：深入 function call（2-3 天）🔧

回到你的日常工作中，用前面学到的整体视角重新审视。

### 学习目标

- 从源码层面理解 function call 的完整实现
- 理解 jinja 模板的作用和渲染时机
- 理解 tool_parser 的实现
- 能定位和修复 function call 相关的问题

### 核心问题

1. OpenAI 的 function calling API 格式是什么样的？vLLM 怎么兼容的？
2. tools 参数怎么被渲染进 prompt 的？jinja 模板在哪？
3. 模型怎么知道应该输出 tool call？停词（stop tokens）怎么控制？
4. tool_parser 解析模型输出时，怎么处理多个 tool call？
5. ascend-vLLM 对这个流程有没有改动？改了什么？

### 代码漫游路径

```
# 先找入口
third_party/vllm/vllm/entrypoints/openai/
  └─ protocol.py                             # API 协议定义
  └─ serving_chat.py                         # chat 处理

# 再找模板和解析
third_party/vllm/vllm/entrypoints/llm.py     # 离线推理也涉及

# 模型侧的 tokenizer
third_party/vllm/vllm/transformers/
  └─ tokenizer.py                            # tokenizer 加载

# ascend 差异
third_party/vllm-ascend/
  └─ (搜索 function_call, tool 相关代码)

third_party/vllm-cloud-main/
  └─ (搜索 patch 中的 function call 改动)
```

### 🧪 动手实验

1. 用一个支持 function calling 的模型，构造一个包含 tools 的请求，抓包看实际发送给模型的 prompt 是什么
2. 修改 tool_parser，加一行日志打印解析结果

### 🗣 费曼检查点

> "一个 function call 请求从 API 进来到响应出去，tools 信息经过了什么变换？jinja 和 tool_parser 分别在什么时候做了什么？"

---

## Phase 5：分布式推理（3-5 天）

### 学习目标

- 理解为什么需要分布式推理（模型太大放不进一张卡）
- 理解 tensor parallelism 和 pipeline parallelism 的区别
- 理解 vLLM 的分布式架构

### 核心问题

1. 一个 70B 模型大概占多少显存？一张 A100 80GB 能放下吗？
2. Tensor parallelism 是怎么把一个 transformer 层切到多张卡上的？
3. Pipeline parallelism 又是什么？它和 tensor parallelism 有什么区别？
4. vLLM 用哪种并行方式？什么时候需要？
5. 多卡通信是怎么做的？（NCCL？CANN 对应的通信库是什么？）

### 学习内容

| 顺序 | 主题 |
|---|---|
| 1 | 模型并行 vs 数据并行 vs 流水线并行 |
| 2 | Tensor parallelism（Megatron-LM 的分片方式） |
| 3 | vLLM 的分布式实现 |

### 代码漫游路径

```
third_party/vllm/vllm/distributed/          # 分布式基础设施
  └─ parallel_state.py                       # 并行状态管理
  └─ comms.py                                # 通信接口

third_party/vllm/vllm/worker/
  └─ worker.py                               # 单卡工作者
  └─ multi_step_worker.py                    # 多卡工作者

third_party/vllm/vllm/config.py
  └─ ParallelConfig                          # 并行配置
```

### 🗣 费曼检查点

> "如果模型太大放不进一张 GPU，怎么用多张卡来跑推理？Tensor parallelism 和 pipeline parallelism 各自怎么切分计算？"

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
