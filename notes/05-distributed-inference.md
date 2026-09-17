# Phase 5：分布式推理 — 知识地图

> 学习日期: 2026-07-15 ~ 2026-08-12
> 状态: 策略层 TP/PP **已复述已验证**（08-12），其余"已读（走过一遍）"，**待用户复述验证**。
> 阅读覆盖: 组拓扑 ✅ | TP 实现 ✅ **已复述** | PP ✅ **已复述** | EP ✅ | AsyncLLM 执行流 ✅ | Mooncake Disagg ✅ | 通信原语（部分）| 配置案例 ✅（概念层，决策链）| ctypes 绑定+枚举翻译 ✅（08-12）
> 结构性缺口: **调度层**（1F1B 交错**无需深学**——vLLM 推理无此机制，概念已覆盖）| 通信层原语已闭环（08-12: ctypes/枚举翻译 + 封装层 + Ring 算法）
> 参考资源: [Ailing Zhang 图解](http://ailzhang.github.io/posts/distributed-compute-in-transformer/) | [王二·并行策略图解](https://wanger-sjtu.github.io/2026-05-11-llm-inference-parallel-strategies/) | [廖维明·llama.py 剖析](https://www.liaoweiming.org/blog/vllm-distributed-inference) | [NVIDIA Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html) | [vLLM 官方博客](https://vllm.ai/blog/2025-02-17-distributed-inference)

---

## 0. 知识地图总览（2026-08-05 记录）

### 分层结构

```
为什么分布式推理
├── 动机层: 单卡放不下(显存墙) / 算不动(带宽墙) / 要吞吐
│
├── 策略层: 切什么?            ← 每种策略解决一个"放不下/算不动"
│   ├── TP  切权重维度(参数)      [已读: 列切/行切 + all-reduce]
│   ├── PP  切层(深度)           [已读: make_layers + isend/irecv]
│   ├── EP  切 expert(MoE)       [已读: SparseMoeBlock + all-to-all]
│   ├── DP  复制模型切 batch      [已读: vLLM 中只为 MoE 服务]
│   ├── SP  切序列(激活)          [未展开]
│   ├── CP  切序列(attention 内)  [未展开]
│   └── 判断标准: 带宽等级决定取舍 (HBM 2TB/s vs NVLink 600GB/s vs IB 25GB/s)
│
├── 通信层: 怎么同步?           ← TP 的灵魂
│   ├── 原语: all-reduce / all-gather / reduce-scatter / all-to-all / p2p
│   ├── Ring 算法: 为什么总流量 2(p-1)/p 倍
│   └── 通信原语细读 communication_op.py   [⚠️ 部分读过, 待细读]
│
├── 调度层: 怎么编排?           ← PP 的灵魂
│   ├── micro-batch 流水线(1F1B)
│   ├── 气泡率公式 (PP-1)/(PP-1+mb)
│   └── 计算/通信重叠(AsyncIntermediateTensors)  [已读]
│
├── 工程层: vLLM 怎么实现?
│   ├── 组拓扑 5D tensor         [已读: parallel_state.py]
│   ├── 多卡执行流 collective_rpc→MQ  [已读]
│   ├── 权重分片 load_weights    [已读: weight_loader/narrow]
│   ├── KV cache 与 TP 的关系    [已读: 按 head 分片]
│   └── 多卡配置案例: 实际部署选 TP/PP/EP  [✅ 概念层: 决策链已推, 见 08-12 记录]
│
└── 进阶/组合
    ├── Disaggregated Prefill/Decode (Mooncake)  [已读]
    ├── 3D 并行组合 TP×PP×DP
    └── 弹性 EP (elastic_ep/)     [❌ 待学]
```

### 状态明细

| 层次 | 状态 | 说明 |
|---|---|---|
| 动机层 | ✅ | 70B=140GB 已算过，带宽墙已理解 |
| 策略层 TP | ✅ **已复述已验证** | 列切/行切、配对省通信、每层2次all-reduce |
| 策略层 PP | ✅ **已复述已验证** | 分层传 hidden_states、isend/irecv、流水线/气泡 |
| 策略层 EP | ✅ 已读待复述 | all-to-all 路由 |
| 策略层 SP/CP | ⚠️ 名词见过 | 只在图解博客里看过，未展开 |
| 通信层 | ✅ **已闭环**（08-12） | 原语语义 + ctypes/枚举翻译 + 封装层 + Ring 算法 2(p-1)/p 已推 |
| 调度层 | ✅ **概念已闭环**（08-12） | 气泡率✅；PP 采样广播+spec 回退✅；1F1B 确认 vLLM 无此机制，无需深学 |
| 工程层 | ✅ 大部分 | 配置案例✅（决策链已推） |
| 进阶 | ⚠️ | Mooncake 读过，elastic_ep 待学 |

### 学习入口建议

1. **通信层深挖**（Ring all-reduce + communication_op.py 细读）→ 补最薄弱的环节（推荐先做）
2. **配置案例**（实际部署怎么选 TP/PP/EP，用图解博客落地）→ ✅ 已做（08-12 决策链），可回头对照博客验证
3. **复述验证**（把已读的 TP/PP 用自己的话讲一遍，确认掌握度再往下走）→ ✅ 已做（08-12，TP/PP 已复述通过），待复述剩余项: EP / AsyncLLM 执行流 / Mooncake

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

### 2026-08-12: 配置决策链（TP×PP×DP 取舍逻辑）

**理解了什么**（从 08-03 的 TP 案例扩展成完整决策链）:
- **PP 的痛 = 流水线气泡**: 流水线作业，前卡算完才能传后卡，算的时候其他卡空闲 → PP 不宜多（气泡率跟 PP 段数、micro-batch 数有关，1F1B 可优化）
- **TP 的痛 = 通信开销**: 分工合作，一次计算就要在参与卡之间通信（all-reduce/all-gather），TP 大 → 通信量大且频率高 → TP 也不宜多
- **两端权衡**: TP大PP小 → 每次计算都高频卡间通信，通信开销大；TP小PP大 → 请求来时前卡计算后卡闲置，计算浪费大
- **决策链（四步）**:
  1. 单卡放得下？→ 放得下: TP=1，直接 DP 多副本（如 7B=14GB）
  2. 放不下 → TP 切权重（列切/行切）
  3. TP 切到极限还装不下 → PP 切层（层间传 hidden_states）
  4. 权重放下后剩的卡能放完整副本 → DP 分担请求（副本间不通信，提吞吐）
  - 特例: EP 只对 MoE 模型存在
- **关键洞见**: DP 在决策链里排最后 = "用剩下的卡"的自然选择；TP/PP 是"塞下权重"的必须手段，DP 是"最大化吞吐"的加分项

**疑问与解答**:

| Q | A |
|---|-----|
| PP 为什么不宜多？ | 流水线气泡：后卡要等前卡算完，等待时计算资源空闲 |
| TP 为什么不宜多？ | 每次计算都要卡间通信，TP 越大通信量/频率越高，带宽成为瓶颈 |
| 为什么是"权衡"而非"取大"？ | 两个维度的痛点互相独立：TP 痛在通信、PP 痛在空闲，只能取中间平衡 |
| DP 为什么排决策链最后？ | 它不解决"放不下"，只解决"吞吐不够"；副本间无通信、可随意复制，是最自然的"剩余卡"利用方式 |

**卡点**: 无（顺着 08-03 的 TP 案例自然延伸，全程是自己推出来的）

**下次入口**: ① 通信层深挖（Ring 算法 + communication_op.py）② 1F1B/气泡率公式细算

---

### 2026-08-12: 复述验证 TP/PP（费曼检查点通过）

**验证方式**: 用户用自己的话复述 TP 链路 + PP 链路，AI 确认/精确化，无大错 → 标记"已复述已验证"

**TP 复述（用户版本）**:
- 切分: w_gate/w_up 列切 + w_down 行切；qkv 列切 + o_proj 列切
- 关键省通信: 列切输出直接喂行切/列切配对（w_up 列切 → w_down 行切；attention 输出 → o_proj 列切），**中间零通信**
- 通信: 每层只需 2 次 all-reduce（FFN 一次 + attention 一次）
- 激活函数（SiLU）逐元素，对分片友好，卡内本地算
- attention 逐 head 独立 → 这是 TP 能切 head 到不同卡的可行性基础
- 洞见: o_proj 输出是"部分和/贡献"（partial contribution），最后 all-reduce 合成——"贡献"用词准确 = 真懂

**PP 复述（用户版本）**:
- 切分: 按 transformer 层切，每卡拥有一部分层（32 层 ÷ PP=4 → 每卡 8 层）
- isend/irecv: 传递**卡内最后一层的输出**（hidden_states），不是权重；只有层边界（7→8 层）跨卡
- 异步点对点: 发方算完立刻丢出不等对方收 → 流水能流起来的通信基础
- 流水线: 卡0 算完立刻接任务2，任务首尾相接，不空等
- 代价: 任务1到达卡3 前卡3 空等 = 气泡

**TP vs PP 本质区别（用户自推）**:

| | TP | PP |
|---|---|---|
| 切什么 | 矩阵（权重维度） | transformer 层（深度） |
| 每卡算 | 同一层的部分（列分片/部分和） | 完整的前 N 层 |
| 通信内容 | 部分和/分片 → all-reduce 合成 | 完整激活（hidden_states 移交） |
| 通信语义 | 算完要合并 | 算完就移交 |

**卡点**: 无。中间省 all-gather 是用户自己推出来的（"这个是你自己搞定的"）

**下次入口**: ① 通信层深挖（Ring 算法 + communication_op.py）② 1F1B/气泡率公式细算

---

### 2026-08-12: 通信层第一块基石 — ctypes 绑定与操作符枚举翻译

**理解了什么**（从 `ReduceOp.SUM: RedOpType = ...` 这个写法切入）:
- `SUM: RedOpType = ...` 是 **.pyi 类型桩**写法：Ellipsis = "值不重要，类型才重要"，是给类型检查器看的接口声明，真身是 C++ 扩展（`torch/_C/_distributed_c10d.pyi:124`）
- **vLLM 的 ncclRedOpTypeEnum 是真值表**（`pynccl_wrapper.py:117`）：`ncclSum = 0` 直接给真实 int
- **ctypes 绑定**: `Function` dataclass（name/restype/argtypes）= C 函数签名表；`ncclRedOp_t = ctypes.c_int`（114 行）告诉 ctypes "这个参数是 int"
- **完整调用链**:
  ```
  调用方: dist.all_reduce(tensor, op=ReduceOp.SUM)   ← torch C++ 枚举对象
    → pynccl.py: all_reduce(op 入参)
    → ncclRedOpTypeEnum.from_torch(op)              ← 翻译: torch语义 → int(ncclSum=0)
    → self.ncclAllReduce(..., op=int, ...)          ← ctypes 按 argtypes[4]=ncclRedOp_t 打包
    → libnccl.so: ncclAllReduce(..., ncclRedOp_t op) ← C 拿到 int 枚举
  ```
- **本质**: Python 世界（ReduceOp 对象）→ ctypes 边界（int）→ C 世界（ncclRedOp_t）；ctypes 是桥，from_torch 是换货币
- **为什么不能 int(op) 硬转**: torch.ReduceOp 和 ncclSum 是两套不同代码的枚举约定，数值不能假设相同，必须显式映射

**疑问与解答**:

| Q | A |
|---|-----|
| `SUM: RedOpType = ...` 三个点啥意思？ | Ellipsis 字面量，stub 里表示"值不重要，类型才重要"——声明存在 + 类型，真值在 C++ |
| 为什么 from_torch 要做转换？ | ctypes 函数签名要求 int（ncclRedOp_t），torch 给的是 C++ 枚举对象，必须翻译 |
| ncclRedOpTypeEnum 为什么有真值？ | 它是 vLLM 自己写的 Python 枚举（真值表），torch 是桩，两者风格不同 |

**卡点**: 无（顺着好奇心菜单的源码问题自然深入，全程自己读代码+自己串链路）

**后续补充（同 session）— 封装层真相**:
- `communication_op.py` 的 5 个函数 = **薄转发器**，每个就一行，无 backend 逻辑:
  ```python
  def tensor_model_parallel_all_reduce(input_):
      return get_tp_group().all_reduce(input_)   # 模型 → TP 组对象
  ```
- **分层真相**:
  ```
  模型层:   tensor_model_parallel_gather(input_, dst)        ← 转发器
  寻址层:   get_tp_group().gather(input_, dst, dim)          ← 拿 TP 组
  实现层:   GroupCoordinator.gather → device_communicator.gather
  翻译层:   dst=self.ranks[dst]                              ← 组内位置 → 全局 rank
  后端:     torch.distributed.gather(dst=全局rank, group=device_group)
  引擎:     libnccl.so
  ```
- **纠偏**: "字符串 nccl"（ray_communicator.py:255 `get_transport_name()`）是**结果汇报**（告诉 Mooncake 用 NCCL 传输），不是后端选择开关；真正的 backend 在 torch.distributed group（`dist.get_backend`），vLLM 侧只有断言校验（pynccl.py:78 等）
- **正反馈**: `self.ranks[dst]` 是用户 08-05 学的，今天无提示复述出来 → 确认真记住

**下次入口**: ① 通信层继续: Ring 算法（all-reduce 为什么 2(p-1)/p 流量）② communication_op.py 已完 ③ 1F1B

---

### 2026-08-12: Ring all-reduce 算法推导（用户自己推出公式）

**理解了什么**（全程用户自推）:
- **核心直觉（用户猜的）**: 每张卡给相邻一个方向的卡传数据，接收方叠加 → 单向环累加
- **关键缺口发现**: 只转一圈 = 只完成 reduce-scatter（每卡拿到自己那份总和），**还差 all-gather 阶段**（把每份总和分发到所有卡）
- **公式推导（用户推的）**:
  ```
  reduce-scatter: 每卡转 p-1 轮 × p 卡 = (p-1)p 次传输
  all-gather:     再 (p-1)p 次
  总传输 = 2(p-1)p 份
  总数据 = p² 份（p 卡 × p 份）
  流量 = 2(p-1)p / p² = 2(p-1)/p
  ```
- **公式物理含义**:
  | 部分 | 含义 |
  |---|---|
  | 2 | 两个阶段（reduce-scatter + all-gather） |
  | p-1 | 环上自己不传给自己，转 p-1 轮 |
  | /p | 数据切 p 份，每轮只传 1/p |
- **边界验证**: p=2 → 1 倍（两卡互传，总数据口径 1/2+1/2）；p→∞ → 2 倍（永不爆炸）
- **对比价值**: 朴素中心化 = (p-1) 倍（每卡直连所有卡，随卡数爆炸）；Ring = 最多 2 倍 → **通信量有界**

**疑问与解答**:

| Q | A |
|---|-----|
| 为什么"转一圈"不够？ | 转一圈只完成 reduce-scatter（每卡只有自己那份总和），要让所有卡都拿到全部总和需要第二个阶段 all-gather |
| p 是什么？ | 参与 all-reduce 的卡数（rank 数） |
| 为什么流量永远 1~2 倍？ | 2(p-1)/p 随 p 增大趋近 2，但永不超 2——这就是 Ring 相对中心化 (p-1) 倍的核心优势 |

**卡点**: 无。初猜"12 次完成"漏了 all-gather 半程——被引导后自己发现缺口并推出完整公式

**下次入口**: ① EP 复述验证 ② SP/CP 展开 ③ 1F1B 具体调度实现（气泡率已推导✅）

---

### 2026-08-12: 流水线气泡率推导（用户自己推出公式）

**理解了什么**（全程用户自推，PP=4 / mb=4 模型）:
- **总耗时** = (PP-1) 启动延迟 + mb 计算 = 3+4 = 7 个时间单位（每卡算一个 mb 用 1 单位）
- **卡3（末卡）空闲** = 前 3 格（等任务流过来）
- **气泡率** = 空闲/总耗时 = 3/7 = **(PP-1)/(PP-1+mb)**
- **气泡的对称性（用户自己发现的漂亮性质）**:
  | 卡 | 启动气泡(head) | 收尾气泡(tail) | 总空闲 |
  |---|---|---|---|
  | 卡0 | 0 | 3 | **3** |
  | 卡1 | 1 | 2 | **3** |
  | 卡2 | 2 | 1 | **3** |
  | 卡3 | 3 | 0 | **3** |
  → 每张卡总空闲都是 PP-1，气泡总量守恒，只是"位置"不同（前卡闲在收尾、后卡闲在启动）→ **气泡是流水线的结构性浪费，不是某张卡的局部问题**
- **mb 的作用**（衔接 1F1B 的钥匙）: 气泡率 = (PP-1)/(PP-1+mb)，mb 越大气泡率越小:
  - mb=1: 3/4 = 75%（灾难）
  - mb=4: 3/7 ≈ 43%
  - mb→∞: → 0（流水线满负荷）
  → **这就是 PP 需要 micro-batch 流水线调度（1F1B）的原因**：用足够多的 mb 把气泡压下去

**疑问与解答**:

| Q | A |
|---|-----|
| 什么是气泡？ | 流水线中卡空闲的时间：启动阶段后卡等任务、收尾阶段前卡等后卡 |
| 气泡率公式怎么来的？ | 总耗时 (PP-1)+mb，空闲 PP-1（每卡总空闲守恒），相除即得 |
| 为什么需要 1F1B？ | mb 越大气泡率越小，1F1B 用微批次交错把流水线填满 |

**卡点**: 无。对称性表格是用户自己排出来的

**下次入口**: ① 1F1B 具体调度（schedule 怎么交错 micro-batch，vllm 代码）② EP 复述验证 ③ SP/CP 展开

---

### 2026-08-12: PP 采样广播与 spec decode 回退（scheduler 层）

**入口**: `vllm/v1/worker/gpu/pp_utils.py`（41 行）+ `model_runner.py` sample_tokens / postprocess + `scheduler.py` 回退逻辑

**理解了什么**:
- **pp_utils.py 是"末卡采样结果广播"，不是层间 isend/irecv**:
  - `pp_broadcast`: 仅末卡调用（`assert is_last_rank`），把 sampled_token_ids + num_sampled/num_rejected 广播给全 PP 组
  - `pp_receive`: 非末卡调用（`assert not is_last_rank`），接收末卡广播
  - **为什么用 broadcast 不是 isend/irecv**: hidden_states 是"只给下一卡"（点对点）；采样结果是"所有卡都要"（调度器跑在每个 rank 上，都要推进 step）
- **num_sampled/num_rejected 的真实语义**（用户直觉对了一半）:
  | 场景 | num_sampled | num_rejected |
  |---|---|---|
  | 普通解码（无 draft）| 1 | 0 |
  | spec decode（提议 N 个）| 接受数（可 >1）| N - 接受数（可同时非零）|
  | chunked prefill | 0 | 0 |
  - `max_sample_len = num_speculative_steps + 1`（model_runner.py:1171）→ spec decode 一次提议多 token 的证据
  - 分支: `num_draft_tokens == 0` → 普通 sampler；否则 → rejection_sampler（890-903）
- **回退（rollback）不是重算**（用户最初的表述被精确化）:
  - scheduler.py:1370-1384: `num_rejected = num_draft_tokens - num_accepted`；`request.num_computed_tokens -= num_rejected`；`num_output_placeholders -= num_rejected`
  - **语义**: 被接受的 draft 正常进 KV cache（不重算），被拒绝的**回退指针**（num_computed_tokens 减回去），下次调度从最后接受的位置继续
  - 被拒绝的 draft 的 KV cache = **无效 spec token**（`num_invalid_spec_tokens`），算了但没用 → 浪费
- **PP 为什么必须广播采样结果**（用户自推）: 调度器跑在每个 rank 上，采样只在末卡 → 所有 rank 都要知道"白算了几格"才能各自更新 KV cache 分配、保持 token 位置一致

**疑问与解答**:

| Q | A |
|---|-----|
| pp_broadcast 传的 sample_token_ids 是采样结果吗？ | 是——末卡最后一层采样出的最终 token |
| 为什么普通解码 num_sampled 永远是 1？ | 每序列每步只采 1 个 token，无 draft |
| 两个计数什么时候同时非零？ | spec decode：提议 4 接受 2 → sampled=2, rejected=2 |
| 被拒绝的 token 怎么处理？ | 不是重算，是回退指针：num_computed_tokens -= num_rejected，KV cache 位置作废 |

**卡点**: 无。用户自己发现"普通解码应该 1/0"的疑点 → 追代码发现 spec decode 场景两个计数可同时非零

**下次入口**: ① 1F1B 具体调度交错（scheduler PP 段 micro-batch 编排）② EP 复述验证 ③ SP/CP 展开

## 目录

0. [知识地图总览](#0-知识地图总览2026-08-05-记录)
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

**列切 vs 行切 → all-gather vs all-reduce（2026-08-05 自己推导，非 AI 讲解）**

为什么不能互换——用 2×2 例子钉死：

```
列切 → all-gather（拼接）
  W 按列切: Y = X·[W_L | W_R]
    卡0: X·W_L = (1,2)  ← Y 的左半段
    卡1: X·W_R = (1,2)  ← Y 的右半段
  拼起来 = (1,4) 完整 Y
  每卡片段互相独立、合起来才完整 → 拼接

行切 → all-reduce（求和）
  W 按行切: Y = [X_T | X_B]·[W_T; W_B]
    卡0: X_T·W_T = (1,4)  ← 完整形状，但是"部分和"
    卡1: X_B·W_B = (1,4)  ← 完整形状，但是"部分和"
  Y = 部分和0 + 部分和1 → 逐元素相加
  每卡都是完整形状向量，但每位置只有部分贡献 → 同位置相加
```

一句话：**all-gather 处理"互不重叠的片段"（拼起来），all-reduce 处理"互相重叠的贡献"（加起来）。语义不同，不能互换。**

### 通信原语细读（2026-08-10 用户读代码闭环 ✅）

`vllm/distributed/communication_op.py`（43 行）——**薄壳转发层**，5 个函数全是转发：

```python
def tensor_model_parallel_all_reduce(input_):
    return get_tp_group().all_reduce(input_)   # 真正的实现在 GroupCoordinator
```

**4 个原语的完整语义**（用户对比推导）:

| 操作 | 数据流向 | 谁拿结果 | 通信量 |
|------|---------|---------|--------|
| all_reduce | 多对多 | 所有人拿完整和 | 2× 数据量（每人拿全量，有 N-1 份浪费） |
| all_gather | 多对多 | 所有人拿完整拼接 | 1× 数据量 |
| reduce_scatter | 多对多 | 每人拿"和的 1/N 块" | 1× 数据量（= reduce+scatter，省 all-reduce 的浪费） |
| gather | **多对一** | **只有 dst 拿完整** | 收集到 dst |

- reduce_scatter 误区：不是"分享给指定的人"，是 **reduce 后按维切块、每人拿自己那块**（all-reduce 的省通信版本）
- gather vs all_gather 只差一个 dst：gather 只有 dst 拿全量，all_gather 所有人拿全量
- 用途：gather 用于 logits 最终收集（`logits_processor.py:86`）——采样只要一份完整 logits，多对一即可；TPU 不支持 gather 原语时退化为 all_gather

**local_rank vs global_rank（用户疑问闭环）**:

```
gather(input_, dst=local_rank)          ← 接口层: 组内语义（"TP 组里第几个"）
  ↓ self.ranks[dst]                     ← 查表: local_rank → global rank
torch.distributed.gather(..., dst=global_rank)   ← 实现层: NCCL 要全局 rank
```

- `self.ranks` = "组内位置 → 全局 rank"映射表（base_device_communicator.py:151/158）
- 接口层故意暴露 local_rank（组内语义）：调用方不需要知道自己在全局是几号，也不需要关心 TP 组跨不跨节点
- dst = "我去哪"（目标位置索引）；rank_in_group = "我是谁"（自己的位置）——语义不同，不能混用

**多层封装链**:
```
communication_op.py (薄壳) → GroupCoordinator (parallel_state.py:1221 get_tp_group)
→ DeviceCommunicatorBase.gather (base_device_communicator.py:255)
→ torch.distributed.gather (NCCL)
```

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
