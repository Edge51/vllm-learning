# Phase 6: Ascend 适配与 CANN — 学习地图

> 学习日期: 2026-08-12 开荒
> 目标: 理解 ascend-vLLM 作为硬件插件的架构、CANN 与 CUDA 的对应关系、能看懂适配代码
> 前提: 已掌握 vLLM GPU 侧架构（worker/model_runner/scheduler/PP/TP/EP/通信层）

---

## 0. 为什么学这个（你的工作动机）

你在 **ascend-vLLM 组维护 function call 特性**。你维护的代码跑在 **NPU（华为昇腾）** 上，不是 CUDA GPU。理解适配层 = 理解你日常打交道的代码的"地基"。

- 你写的 function call 相关代码，哪些是平台无关的？哪些在 NPU 上有差异？（采样、logits processor 与平台关系不大；attention、通信、算子调用差异大）
- 出了性能/正确性问题，能不能判断是"vLLM 逻辑问题"还是"NPU 适配问题"？

---

## 1. 分层总览（先有地图，再读代码）

```
┌─────────────────────────────────────────────────────┐
│  你的代码（function call / 模型逻辑 / scheduler）    │ ← 平台无关
├─────────────────────────────────────────────────────┤
│  vLLM 抽象层（Platform / WorkerFactory / Attention  │
│   Backend / 通信封装）                               │ ← 插件挂载点
├─────────────────────────────────────────────────────┤
│  vllm-ascend 插件（NPUPlatform + 各 Backend 实现）   │ ← 你要学的
├─────────────────────────────────────────────────────┤
│  torch_npu（torch 的 NPU 后端，设备名 'npu'）        │
├─────────────────────────────────────────────────────┤
│  CANN（AscendCL / GE / HCCL / 算子库）              │ ← 对标 CUDA 全家桶
├─────────────────────────────────────────────────────┤
│  昇腾硬件（AI Core: 向量单元 + 矩阵单元( Cube )）    │ ← 对标 SM
└─────────────────────────────────────────────────────┘
```

**一句话**：vLLM 定义插槽（Platform/Backend），vllm-ascend 往插槽里填 NPU 实现，最终通过 torch_npu + CANN 驱动昇腾硬件。

---

## 2. CANN vs CUDA 对照表（核心心智模型）

| CUDA 概念 | CANN 对应 | 说明 |
|---|---|---|
| CUDA 设备 (`cuda:0`) | NPU 设备 (`npu:0`) | torch_npu 提供的设备后端 |
| CUDA runtime API | **AscendCL (ACL)** | C 接口，设备管理/内存/算子执行 |
| NVIDIA 驱动 | 昇腾驱动 + CANN toolkit | 类似但独立 |
| cuDNN/cuBLAS | **CANN 算子库**（aclnn 算子） | 预编译算子调用接口 |
| NCCL | **HCCL**（华为集合通信库） | all-reduce/all-gather/p2p 都有 |
| nvcc / CUDA kernel | **Ascend C / 算子编译器** | 写 NPU kernel 的语言 |
| SM（流多处理器） | **AI Core** | 昇腾的计算核心 |
| CUDA 核心 | 向量单元 + **Cube 矩阵单元** | 矩阵乘靠 Cube |
| Nsight / ncu | **Ascend 开发工具**（msprof 等） | 性能分析 |

**关键差异点**（写笔记时要懂的）：
1. **矩阵乘单元（Cube）**：昇腾的矩阵算力在 Cube 单元，向量单元管非矩阵算子——所以算子的"形状"适配很重要，GEMM 要切成 Cube 友好的分块
2. **算子编译**：CANN 算子可以预编译（aclnn）或在线编译（GE 图模式），vLLM-ascend 大量用**预编译自定义算子**（csrc/ 里）
3. **HCCL 对标 NCCL**：通信语义一致，但实现独立（vllm_ascend/distributed/ 里有自定义通信器）

---

## 3. 插件机制（vLLM 怎么"插"进 NPU 支持）

### 3.1 注册入口（已读，setup.py:540-548）

```python
entry_points={
    "vllm.platform_plugins": ["ascend = vllm_ascend:register"],       # 平台
    "vllm.general_plugins": [
        "ascend_kv_connector = vllm_ascend:register_connector",      # KV 传输
        "ascend_model_loader = vllm_ascend:register_model_loader",   # 模型加载
        "ascend_service_profiling = vllm_ascend:register_service_profiling", # 性能
        "ascend_model = vllm_ascend:register_model",                 # 模型注册
    ],
}
```

### 3.2 注册函数干的事（已读，__init__.py:38-73）

| 函数 | 作用 |
|---|---|
| `register()` | 返回 `vllm_ascend.platform.NPUPlatform` → vLLM 用这个类做平台判断 |
| `register_connector()` | 注册 KV 传输连接器（对标 Mooncake 那套） |
| `register_model_loader()` | netloader + rforkloader（NPU 专属模型加载） |
| `register_service_profiling()` | 服务级性能配置 |
| `register_model()` | 注册 NPU 上跑的模型实现 |
| `_ensure_global_patch()` | **全局 patch**：engine-core 子进程也生效（scheduler 等代码的 NPU 适配） |

### 3.3 NPUPlatform（platform.py:97）

继承 vLLM 的 `Platform` 抽象类，实现：
- `get_device_capability`（197 行）→ NPU 算力
- `get_device_name/uuid`、`num_compute_units` → 硬件信息
- `set_device`、`inference_mode` → 设备操作
- `check_and_update_config`（276 行）→ 配置校验
- `get_pass_manager_cls`、`get_compile_backend`（121/129 行）→ **编译后端选择**（NPU 用 torch_npu 编译路径）

---

## 4. vllm_ascend 目录地图（怎么读）

```
vllm_ascend/
├── platform.py                 # NPUPlatform（第 3.3 节）— 入口
├── __init__.py                 # 插件注册（第 3.2 节）— 入口
├── worker/
│   ├── worker.py               # NPU worker（对标 gpu_worker.py）
│   ├── model_runner_v1.py      # NPU model runner v1
│   ├── npu_input_batch.py      # NPU 输入批处理
│   ├── block_table.py          # block table（NPU 版）
│   └── v2/                     # model runner v2 适配
├── attention/                  # ★ 重点：7 种 attention 后端
│   ├── abstract.py             # 抽象基类
│   ├── fa3_v1.py               # FlashAttention3
│   ├── mla_v1.py               # Multi-head Latent Attention（DeepSeek）
│   ├── dsa_v1.py / sfa_v1.py   # 其他 NPU 注意力
│   └── context_parallel/       # 上下文并行
├── distributed/
│   ├── device_communicators/   # NPU 通信器（HCCL 封装）
│   ├── parallel_state.py       # 并行状态（NPU 版）
│   └── kv_transfer/            # KV 传输
├── ops/                        # ★ 重点：NPU 算子封装
│   ├── fused_moe/              # MoE 算子（对标 GPU fused_moe）
│   ├── rotary_embedding.py     # RoPE
│   ├── mla.py                  # MLA 算子
│   ├── triton/                 # triton 算子
│   └── register_custom_ops.py  # 自定义算子注册
├── model_loader/               # netloader / rforkloader
├── models/                     # NPU 模型实现（register_model）
├── quantization/               # 量化（A16W8 等）
├── compilation/                # 编译 pass
├── kv_offload/                 # KV cache 卸载
├── spec_decode/                # 投机解码（NPU 版）
├── sample/                     # 采样（NPU 版）
├── lora/                       # LoRA 适配
└── csrc/                       # C++/CANN 自定义 kernel
    ├── attention/              # C++ attention kernel
    ├── moe/                    # C++ MoE kernel
    ├── gmm/                    # grouped mm
    ├── mc2/                    # all-to-all + GEMM 融合
    └── mla_preprocess/         # MLA 预处理
```

---

## 5. 学习路径（建议 3-5 天，一天一块）

### Day 1: 插件机制 + 平台层
- 读 `setup.py` entry_points + `__init__.py` register 函数（已列）
- 读 `platform.py` 的 `NPUPlatform`，对照 vLLM 的 `Platform` 抽象类找差异
- **目标**: 能说出"vLLM 怎么发现 NPU 平台"

### Day 2: Worker + Model Runner（跟 function call 最相关）
- 读 `worker/worker.py` + `model_runner_v1.py`
- 对照 GPU 侧 `gpu_worker.py` / `model_runner.py` 找差异点
- **目标**: 能说出"NPU worker 相比 GPU worker 改了什么"

### Day 3: Attention + 算子层（性能核心）
- 读 `attention/abstract.py` + 一个具体后端（fa3_v1 或 mla_v1）
- 读 `ops/register_custom_ops.py` + 一个算子（rotary_embedding）
- **目标**: 能说出"NPU 的 attention 后端怎么注册、算子怎么封装"

### Day 4: 通信 + 分布式
- 读 `distributed/device_communicators/`（HCCL 封装，对标之前学的 NCCL）
- 对比 GPU 的 `pynccl` / `custom_all_reduce`
- **目标**: 能说出"HCCL 封装 vs NCCL 封装的差异"

### Day 5: 全局 patch + 收尾
- 读 `utils.py` 的 `adapt_patch`（全局 patch 都 patch 了什么）
- 跑通一个示例（examples/），对照 GPU 跑一遍看差异
- **费曼检查点**: "vLLM 在 GPU 跑 vs NPU 跑，架构上有什么区别？CANN 的角色是什么？"

---

## 6. 好奇心菜单（随时可做支线）

| # | 类型 | 问题 |
|---|------|------|
| 1 | 机制型 | vLLM 的 `Platform` 抽象类设计成什么样？为什么用类属性而不是函数注册？ |
| 2 | 对比型 | NPU worker 和 GPU worker 的差异集中在哪些方法？为什么？ |
| 3 | 机制型 | `adapt_patch` 全局 patch 都 patch 了什么？为什么不改 vLLM 源码而用 patch？ |
| 4 | 源码型 | `attention/` 里 7 种后端各是什么模型/场景用的？ |
| 5 | 对比型 | HCCL 封装 vs NCCL 封装（pynccl）的代码差异在哪？ |
| 6 | 改造型 | 如果我来给一个 vLLM 模型写 NPU 适配，要动哪几个文件？ |
| 7 | 机制型 | function call（你的工作）在 NPU 上有什么平台相关的地方？ |

---

## 7. 面试话术沉淀（学完回填）

- [ ] "vLLM 的硬件插件机制是什么？" → entry_points + Platform 抽象
- [ ] "CANN 和 CUDA 怎么对应？" → 对照表（第 2 节）
- [ ] "vllm-ascend 替换了 vLLM 的哪些组件？" → 平台/worker/attention/算子/通信
- [ ] "NPU 的性能瓶颈一般在哪？" → attention 后端、算子形状、HCCL 通信

---

## 8. 状态追踪

- [ ] Day 1 插件机制 + 平台层
- [ ] Day 2 Worker + Model Runner
- [ ] Day 3 Attention + 算子层
- [ ] Day 4 通信 + 分布式
- [ ] Day 5 全局 patch + 费曼复述
