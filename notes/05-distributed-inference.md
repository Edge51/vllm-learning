# Phase 5：分布式推理 — 知识地图

> 学习日期: 2026-07-15
> 状态: 组拓扑 ✅ | TP 实现 ✅ | PP / 通信原语 / EP / Worker / 配置案例 ❌

---

## Session 笔记

### 2026-07-15: 分布式基础概念 + 组拓扑 + TP 列切/行切

**理解了什么**:
- vLLM 把 GPU 排成 5D tensor `(ExternalDP, DP, PP, PCP, TP)`，通过 reshape + unbind 创建各维度的 ProcessGroup
- TP 列切（ColumnParallelLinear）→ GeLU（逐元素）→ TP 行切（RowParallelLinear + all-reduce）的组合省掉一次通信
- DP 在 vLLM 中专用于 MoE expert 切分，非 MoE 模型 DP=1
- MoE 的 all-to-all 通信把 token 路由到 expert 所在 rank

**疑问与解答**:

| Q | A |
|---|-----|
| `reshape(-1, ...)` 的 `-1` 是什么意思？ | 自动推导维度大小，总元素数 ÷ 其他维度积 |
| `unbind(dim)` 是做什么的？ | 沿指定维度把 tensor 拆成多个低一维的 tensor |
| ExternalDP 是什么？ | 独立模型副本数，各副本独立处理请求互不等待 |
| 为什么一张卡不够时要用 TP×PP×DP×PCP 张卡？ | 这些维度的 rank 每步 forward 都需要同步通信，少一张就死锁 |
| Expert 是什么？ | MoE 模型里的多个 FFN，router 选 top-K 激活 |
| GeLU 在 TP 组合里起什么作用？ | 逐元素激活函数，不改变矩阵形状和 shard 方式，不引入通信 |
| 为什么列切→行切组合省通信？ | 列切输出列维度 = 行切输入行维度，shape 天然对齐，不需要 all-gather |
| 列切和行切各自对应什么通信操作？ | 列切→all-gather（或保持 shard），行切→all-reduce |
| reduce-scatter 对比 all-reduce 省在哪？ | 省一半通信量：求和后不广播完整结果，各卡只拿自己那份 |
| TP 大好还是小好？ | 刚好装下模型就行，不要多。TP 越大通信开销越大 |
| DP 值是不是由 TP 决定的？ | 不是。TP 解决 attention/dense 权重太大，DP 解决 expert 太多，两个独立维度 |
| 非 MoE 模型 DP 有意义吗？ | 没有。vLLM 的 DP 只服务 MoE expert 切分，非 MoE 模型 DP=1 |
| 为什么 DeepSeek V3 用 TP=2, DP=4？ | Shared attention 37B 单卡装不下→TP=2；256 expert 太多放不下→DP=4 |
| 70B (FP16 ≈ 140GB) 是什么意思？ | 70Billion 参数，每个参数 FP16=2bytes，70×10⁹×2 ≈ 140GB |

**下次入口**: `init_model_parallel_group` 内部实现 → 看 rank list 怎么变成 `torch.distributed.ProcessGroup`

---

## 目录

1. [组拓扑（5D tensor 分组）](#1-组拓扑5d-tensor-分组)
2. [Tensor Parallel（TP）](#2-tensor-paralleltp)
3. [Pipeline Parallel（PP）](#3-pipeline-parallelpp)
4. [通信原语](#4-通信原语)
5. [Expert Parallel / MoE 路由](#5-expert-parallel--moe-路由)
6. [Worker 架构](#6-worker-架构)

---

## 1. 组拓扑（5D tensor 分组）

### 核心代码

`vllm/distributed/parallel_state.py:1486` `initialize_model_parallel()`

### 5D 张量

```python
all_ranks = torch.arange(world_size).reshape(
    -1,                                      # ExternalDP（独立副本数，自动推导）
    data_parallel_size,                       # DP（MoE Expert Parallelism）
    pipeline_model_parallel_size,             # PP（Pipeline）
    prefill_context_model_parallel_size,      # PCP（Prefill Context）
    tensor_model_parallel_size,               # TP（Tensor Parallel）
)
```

### 各维度含义

| 维度 | 切什么 | 通信方式 | 不能少的原因 |
|------|--------|----------|------------|
| **TP** | 一个 transformer 层的参数（按 head / 列切） | all-reduce, all-gather, reduce-scatter | 每卡只持有一部分权重，必须全部合起来才算完 |
| **PP** | 层（rank 0 前 8 层，rank 1 后 8 层） | p2p send / recv | 前向是流水线，中间断了传不下去 |
| **PCP** | Prefill 阶段的 KV cache 计算 | all-reduce | prefill 注意力计算分散在各卡上，最后要合并 |
| **DP** | MoE 专家参数（每卡只持有一部分 expert） | all-to-all | token 需要路由到 expert 所在 rank |
| **ExternalDP** | 完整模型副本（水平扩缩） | 不需要 | 各副本独立处理请求，互不等待 |

### group 创建方法

```python
# 从 all_ranks 中提取某维度的 rank 列表，创建 ProcessGroup
group_ranks = all_ranks.reshape(-1, tp_size).unbind(0)
group_ranks = [x.tolist() for x in group_ranks]
_TP = init_model_parallel_group(group_ranks, ...)
```

- `-1` 在 reshape 中表示自动推导（总元素数 ÷ 其他维度积）
- `unbind(0)` 把 2D tensor 按第一维拆成列表

### 配置约束

- **显存**: TP × PP × PCP × DP 张卡必须能装下模型权重 + KV cache
- **整除**: TP 必须是注意力头数的因数
- **通信带宽**: TP 需要 NVLink（600GB/s+）；PP 可以用 RDMA（200GbE）；DP 通信量最小

### 实际案例

```
8 × A100 80GB, 70B 模型
TP=4（每卡 ~35GB 权重 + KV cache）→ 每副本 4 卡
8 ÷ 4 = 2 个副本 → ExternalDP=2
reshape(-1, DP=1, PP=1, PCP=1, TP=4) → 自动推出 ExternalDP=2

8 × A100 80GB, 7B 模型
TP=1 → 每副本 1 卡
8 ÷ 1 = 8 个副本 → ExternalDP=8
reshape(-1, DP=1, PP=1, PCP=1, TP=1) → 自动推出 ExternalDP=8
```

---

## 2. Tensor Parallel（TP）

### 核心思想

把矩阵乘法按维度切开到多张卡上并行算，每步 forward 需要通信合并。

### 列切（ColumnParallelLinear）

`vllm/model_executor/layers/linear.py:410`

```python
Y = XA, A = [A₁ | A₂ | ... | Aₚ]
# 每卡: X·A_chunk → (batch, output_size/tp)  形状
```

- `output_size_per_partition = output_size / tp_size`
- `gather_output=False`（默认）→ 不 all-gather，保持 shard 状态给下一层
- 用于：QKV 投影、FFN 第一层

### 行切（RowParallelLinear）

`vllm/model_executor/layers/linear.py:1389`

```python
Y = [X₁ | X₂ | ... | Xₚ]·B
# 每卡: X_chunk·B → 部分和 → all-reduce 合并
```

- `input_is_parallel`: 如果输入已经是 shard 好的（来自 ColumnParallelLinear），跳过拆分
- all-reduce 收尾得到完整结果
- 用于：Attention 输出投影、FFN 第二层

### 省通信的关键组合

```
列切 A → GeLU → 行切 B → all-reduce
├── 0 通信 ─┤ (shape 天然对齐, GeLU 是逐元素不变形状)
            └────────── 只此一次通信 ──────────┘
```

### GeLU

- Gaussian Error Linear Unit，Transformer FFN 的非线性激活函数
- 逐元素操作，不需要完整矩阵
- 没有它，两层线性就等效于一层（非线性打破合并性）
- 放在列切和行切之间，不引入额外通信

### 通信操作对比

| 操作 | 通信量 | 用途 |
|------|--------|------|
| all-reduce | 2× 数据量 | 行切后合并部分和 |
| reduce-scatter | 1× 数据量 | 各卡求和后每人拿自己那份（序列并行） |
| all-gather | 1× 数据量 | 列切后拼接完整输出 |

### 相关代码

```python
# ColumnParallelLinear 初始化
self.output_size_per_partition = divide(output_size, self.tp_size)

# RowParallelLinear forward
# 1. 拆分输入（如果 input_is_parallel=False）
split_input = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)
input_parallel = split_input[self.tp_rank]
# 2. 矩阵乘法
output_parallel = self.quant_method.apply(self, input_parallel, bias_)
# 3. all-reduce 合并
output = tensor_model_parallel_all_reduce(output_parallel)
```

---

## 3. Pipeline Parallel（PP）

> ❌ 还没看

### 待学习内容

- 层级别切分方式
- p2p send / recv 传输激活值
- 微批次（micro-batch）调度
- PP 与 TP 的组合

---

## 4. 通信原语

> ❌ 还没看

### 待学习内容

- `vllm/distributed/communication_op.py`
- all-reduce / all-gather / reduce-scatter 包装
- broadcast / send / recv
- 每种操作用在哪个并行维度

---

## 5. Expert Parallel / MoE 路由

> ❌ 还没看

### 待学习内容

- DP 与 EP 的关系
- all-to-all 通信（token 路由到 expert 所在 rank）
- DeepSeek V3 256 expert 的切分方式
- `vllm/distributed/elastic_ep/`
- Expert 并行负载均衡（EPLB）

---

## 6. Worker 架构

> ❌ 还没看

### 待学习内容

- `vllm/worker/worker.py`（单卡工作者）
- `vllm/worker/multi_step_worker.py`（多步 worker）
- executor（调度 worker 执行）

---

## 相关文件索引

```
# 分布式基础设施
third_party/vllm/vllm/distributed/
├── parallel_state.py           # GroupCoordinator, initialize_model_parallel
├── communication_op.py         # all-reduce / all-gather 等通信封装
├── device_communicators/       # 设备通信器（NCCL, CANN 等）
├── elastic_ep/                 # 弹性 Expert Parallel
├── utils.py                    # StatelessProcessGroup

# 配置
third_party/vllm/vllm/config/parallel.py
└── ParallelConfig              # 所有并行维度配置

# 模型层
third_party/vllm/vllm/model_executor/layers/linear.py
├── ColumnParallelLinear        # 列切线性层
├── MergedColumnParallelLinear  # 合并列切（QKV 等）
└── RowParallelLinear           # 行切线性层

# Worker
third_party/vllm/vllm/worker/
├── worker.py                   # 单卡工作者
└── multi_step_worker.py        # 多步工作者

# Executor
third_party/vllm/vllm/v1/executor/  # 调度 worker 执行
```
