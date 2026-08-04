# Phase 5：分布式推理 — 知识地图

> 学习日期: 2026-07-15 ~ 2026-08-02
> 状态: 全部为"已读（走过一遍）"，**待用户复述验证**。掌握程度未确认，勿当作已学。
> 阅读覆盖: 组拓扑 ✅ | TP 实现 ✅ | PP ✅ | EP ✅ | AsyncLLM 执行流 ✅ | Mooncake Disagg ✅ | 通信原语（部分）| 配置案例 ❌

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

### 2026-08-02: 模型层并行实现（Qwen2 源码）

**理解了什么**:
- Qwen2Model 骨架：embed_tokens → DecoderLayer×N → norm → hidden_states
- Qwen2DecoderLayer：input_layernorm → attention → post_attention_layernorm → MLP（residual 融合在 RMSNorm 里）
- Qwen2MLP：MergedColumnParallelLinear(hidden→gate+up) → SiluAndMul → RowParallelLinear(all-reduce)
- ColumnParallelLinear：output_dim（dim=0）切分，weight_loader 用 `narrow(output_dim, tp_rank * shard_size, shard_size)`
- RowParallelLinear：input_dim（dim=1）切分，all-reduce 合并部分和
- QKVParallelLinear：按 head 级别切分（不是连续 dim），保持 head 完整性
- RMSNorm：per-token 沿 hidden_size 归一化；residual 融合 `x + residual`
- SiluAndMul：`silu(x[:d]) * x[d:]`（SwiGLU 门控），不是 GeLU
- Attention backend 选择：platform.get_valid_backends → 优先级列表 → 第一个通过 validate_configuration 的

**疑问与解答**:

| Q | A |
|---|-----|
| （待补） | |
| （待补） | |

**下次入口**: （待补）——例如继续看某层的 forward 细节，或跳去别处

---

### 2026-08-02: PP（Pipeline Parallel）+ AsyncLLM 多卡执行流

**理解了什么**:
- PP 分层：`make_layers` + `get_pp_indices` 把层均分给各 PP rank
- PP 通信：`isend_tensor_dict` / `irecv_tensor_dict`，只传 `IntermediateTensors({"hidden_states", "residual"})`——**不传 KV cache**
- 每个 rank 管自己那几层的 KV cache（物理上在不同 GPU）
- `AsyncIntermediateTensors`：懒等待 comm，实现计算/通信重叠
- 多卡执行流：EngineCore.step() → Scheduler.schedule() → Executor.execute_model() → collective_rpc("execute_model", ...) → rpc_broadcast_mq.enqueue → worker 进程 dequeue → Worker.execute_model() → model_runner
- WorkerBase.model_runner 是 nn.Module 槽位，GPUWorker 赋值为 GPUModelRunnerV1

**疑问与解答**:

| Q | A |
|---|-----|
| （待补） | |
| （待补） | |

**下次入口**: （待补）

---

### 2026-08-02: EP（Expert Parallel）+ Mooncake（Disaggregated Prefill/Decode）

**理解了什么**:
- MoE：FFN 换成 SparseMoeBlock（Router → all-to-all dispatch → expert compute → all-to-all combine），attention/KV cache 不变
- EP vs PP 的区别：PP 按层切（传 hidden_states），Disagg 按阶段切（传完整 KV cache，两个实例都有完整模型）
- Mooncake 架构：Scheduler 端（get_num_new_matched_tokens → 问远端算了多少 token）+ Worker 端（start_load_kv / send_kv_to_decode）
- MooncakeConnector 不逐层同步（save_kv_layer = pass），而是异步批量：register_kv_caches 注册 GPU 显存地址 → ZMQ 协商 → RDMA GPU-to-GPU 直传
- 完整链路：scheduler.py:621 → get_num_new_matched_tokens → update_state_after_alloc(783) → build_connector_meta(960) → SchedulerOutput → kv_connector_model_runner_mixin.py:102 start_load_kv → MooncakeConnectorWorker → send_kv_to_decode → batch_transfer_sync_write

**疑问与解答**:

| Q | A |
|---|-----|
| （待补） | |
| （待补） | |

**下次入口**: （待补）——如 Mooncake 细节 / EPLB 负载均衡

---

### 2026-08-03: TP（Tensor Parallel）多卡配置案例分析

**理解了什么**:
- TP主要是在切分权重，通过all reduce和all gather，来分摊显存的压力
- TP不是越小越好也不是越大越好，而是需要权衡权重的大小和硬件的关系，要考虑权重按照TP大小切分之后，在单节点的卡上面占用了多大比例的空间，然后还要考虑TP如果比较大，则卡间通信的代价也会随之变大，所以需要两方面的权衡


**疑问与解答**:

| Q | A |
|---|-----|
| 70B fp16 权重 140GB 怎么算的？| 700亿参数，7*10^10 * 2 大约140GB|
| TP 为什么不是越大越好？ |因为TP越大 也就是一个权重要在更多的卡上有部分，那计算时all reduce和all gather需要在卡间传递数据的次数和量都会增大，有可能因为通信的带宽和速率而性能遇到瓶颈|
| TP=2/4/8 怎么选、为什么选 4？ |2 的话kv cache只有10GB 太小了，8的话通信带宽和频率会变大，性能受影响|
| 为什么 TP 不跨机器（带宽墙）？RDMA 用在哪？ |TP需要每次计算都做all gather all reduce, 需要大量的通信，如果跨机器，那么通信的速度相比于卡间通信更加慢，性能瓶颈容易受限，RDMA主要用在跨机器的PP/DP/EP/KV cache传输|

**卡点**: 1. 一开始不知道看什么，入口太宽没方向；2. "单节点上限" 是带宽墙不是显存墙，一开始理解偏了

**下次入口**: （待补）——如 PP是如何做的 有什么作用 


---

## 目录

1. [组拓扑（5D tensor 分组）](#1-组拓扑5d-tensor-分组)
2. [Tensor Parallel（TP）](#2-tensor-paralleltp)
3. [Pipeline Parallel（PP）](#3-pipeline-parallelpp)
4. [通信原语](#4-通信原语)
5. [Expert Parallel / MoE 路由](#5-expert-parallel--moe-路由)
6. [Worker 架构](#6-worker-架构)
7. [AsyncLLM 多卡执行流](#7-asyncllm-多卡执行流)
8. [Disaggregated Prefill/Decode（Mooncake）](#8-disaggregated-prefilldecode-mooncake)

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

> ✅ 已读（2026-08-02），细节见 Session 笔记 + 07-interview-prep #11

### 核心要点

- 层切分：`make_layers` + `get_pp_indices`（models/utils.py:620）把层均分给 PP rank
- 通信：`isend_tensor_dict` / `irecv_tensor_dict`（parallel_state.py:851/946）
- **传什么**：只传 `IntermediateTensors({"hidden_states", "residual"})`，不传 KV cache
- 每 rank 管自己那几层的 KV cache（物理上不同 GPU）
- `AsyncIntermediateTensors`（gpu_worker.py:73）：懒等待 comm，计算/通信重叠

### 待学习内容

- 微批次（micro-batch）调度
- PP 与 TP 的组合

---

## 4. 通信原语

> ⚠️ 部分：collective_rpc / MQ 已读；communication_op.py 的 all-reduce 包装未细读

### 待学习内容

- `vllm/distributed/communication_op.py`
- all-reduce / all-gather / reduce-scatter 包装
- broadcast / send / recv
- 每种操作用在哪个并行维度

---

## 5. Expert Parallel / MoE 路由

> ✅ 已读（2026-08-02，EP 概念 + SparseMoeBlock 入口），细节见 Session 笔记

### 核心要点

- MoE 把 FFN 换成 SparseMoeBlock：Router(`gate`) → all-to-all dispatch → expert compute → all-to-all combine
- 只有 FFN 用 expert，attention/KV cache 不变
- 每个 expert 是小型 MLP（gate_up → silu(gate)*up → down）
- 相关文件：qwen2_moe.py:125（SparseMoeBlock）、moe_runner.py:567-767（MoERunner）

### 待学习内容

- `FusedMoE`（fused_moe/layer.py）→ `MoERunner._forward_impl`（moe_runner.py:717）
- DP 与 EP 的关系（DP 即 EP 的另一种叫法）
- DeepSeek V3 256 expert 的切分方式
- `vllm/distributed/elastic_ep/`
- Expert 并行负载均衡（EPLB）

---

## 6. Worker 架构

> ✅ 已读（2026-08-02，Worker/WorkerBase/GPUWorker 结构），细节见 §7

### 核心要点

- WorkerBase：model_runner 是 nn.Module 槽位（worker_base.py:88），load_model/execute_model 抽象
- WorkerWrapperBase：每个 executor 进程一个，`init_worker` 懒初始化
- GPUWorker（gpu_worker.py:105）：execute_model → PP recv(非首 rank) → model_runner.execute_model → PP send(非末 rank)

### 待学习内容

- executor 详细实现（MultiprocExecutor 内部结构）

---

## 7. AsyncLLM 多卡执行流

> ✅ 已读（2026-08-02）

### 完整调用链

```
AsyncLLM
  → EngineCoreClient (IPC)
    → EngineCore.step()（主循环）
      → scheduler.schedule()                    → SchedulerOutput
      → executor.execute_model(scheduler_output)
        → MultiprocExecutor.execute_model()     （multiproc_executor.py:306）
          → collective_rpc("execute_model", args=(scheduler_output,))   （line 339）
            → rpc_broadcast_mq.enqueue(...)     ← 方法名+参数进共享内存 MQ
              ↓ (worker 进程 dequeue)
            → Worker.execute_model(scheduler_output)   ← GPUWorker
              → model_runner.execute_model(...)        ← model forward
      → sampler.sample()                        → 采样 token
      → 更新 request 状态 + 回收 KV cache block
```

### 关键代码

- `multiproc_executor.py:306` `execute_model` → `collective_rpc`
- `multiproc_executor.py:339` `collective_rpc`：`rpc_broadcast_mq.enqueue((method, args, kwargs, output_rank))`
- `worker_base.py:88` `self.model_runner: nn.Module | None`（槽位）
- `worker_base.py:130` `load_model()` 抽象 → GPUWorker 实现 → model_runner.load_model()
- `gpu_worker.py:753` `execute_model`：PP recv → forward → PP send

---

## 8. Disaggregated Prefill/Decode（Mooncake）

> ✅ 已读（2026-08-02），细节见 Session 笔记 + 07-interview-prep #14

### PP vs Disagg 区分（易混点）

| | PP (Pipeline Parallel) | Disaggregated (Mooncake) |
|---|---|---|
| **切什么** | 按层切模型 | 按请求阶段切 |
| **实例关系** | stage 串行，各持部分层 | 两个完整模型实例 |
| **传什么** | hidden_states（中间激活） | **完整 KV cache** |
| **KV cache** | 各 stage 管自己那几层 | prefill 算完整 KV → RDMA 传 decode |

### 架构

- 配置：`KVTransferConfig`（kv_role: producer/consumer/both）
- 抽象：`KVConnectorBase_V1` → Scheduler 端 + Worker 端
- MooncakeConnector：**不逐层同步**（save_kv_layer = pass），异步批量
- 传输：register_kv_caches 注册 GPU 显存地址 → ZMQ 协商 → Mooncake TransferEngine `batch_transfer_sync_write` RDMA 直传 GPU→GPU

### 完整链路（decode 视角）

```
scheduler.py:621  get_num_new_matched_tokens()   问"prefill 算了多少 token？"
scheduler.py:783  update_state_after_alloc()     记录哪些 block 要拉
scheduler.py:946  _build_kv_connector_meta()     → SchedulerOutput.kv_connector_metadata
  ↓ (collective_rpc 到 worker)
kv_connector_model_runner_mixin.py:102  start_load_kv()
  → MooncakeConnectorWorker._start_load_kv()     decode 端
      → receive_kv() → receive_kv_from_single_worker()   ZMQ 请求 prefill
  → MooncakeConnectorWorker.record_send_reqs()   prefill 端
      → send_kv_to_decode()                     响应
          → _build_transfer_params() → _send_blocks()
              → engine.batch_transfer_sync_write(remote, src_ptrs, dst_ptrs, lengths)
```

### 关键文件

- `kv_connector/v1/base.py`：KVConnectorBase_V1、KVConnectorRole（SCHEDULER/WORKER）
- `kv_connector/v1/mooncake/mooncake_connector.py`：MooncakeConnectorScheduler(461)、MooncakeConnectorWorker(703)、send_kv_to_decode(978)、register_kv_caches(1351)、_send_blocks(1332)
- `kv_connector/utils.py:425`：TransferTopology
- `config/kv_transfer.py`：KVTransferConfig

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
