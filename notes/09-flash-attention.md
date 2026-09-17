# FlashAttention — 学习笔记

> 状态: **已讨论（2026-08-05 白天）+ 2026-08-10 vLLM 代码链路 + block_size 闭环**。核心理论已复述确认。
> 铁律: 未复述 = 未学过。本节内容由用户复述确认后才算"已学"。

---

## Session 记录

### 2026-08-10: vLLM 代码链路追踪（从模型入口 → FlashAttention 内核）

**背景**: 用户追 `self.model` 链路 → 模型加载 → Qwen2ForCausalLM → Qwen2Attention → Attention → `torch.ops.vllm.unified_attention_with_output` → backend → Triton kernel，一路戳到 FA 执行层。

**理解了什么**（用户读代码确认）:
- `self.model` 两条链: 启动时 `load_model()` 赋值（gpu_model_runner.py:4791-4792），推理时 `_model_forward` 调用（3524 行）——同一个字段
- 模型解析链: `get_model_loader` → `DefaultModelLoader` → `_get_model_architecture`（utils.py:175）→ `resolve_model_cls`（registry.py:1161）→ `_MODELS["Qwen2ForCausalLM"] = ("qwen2", "Qwen2ForCausalLM")`（registry.py:197）→ 懒加载 → qwen2.py 的 `Qwen2ForCausalLM`（527 行）
- attention 执行链: `Qwen2Attention.forward`（qwen2.py:209）→ `Attention.forward`（attention.py:521）→ `torch.ops.vllm.unified_attention_with_output`（attention.py:754，PyTorch 自定义算子，**实现在 vLLM 内**: csrc/ + triton）→ `self.impl.forward`（backend 分发）→ Triton/CUDA kernel
- backend 注册/选择机制: `AttentionBackendEnum`（registry.py:44，存字符串路径）→ `get_attn_backend`（selector.py:53）→ `current_platform.get_attn_backend_cls`（selector.py:116）→ CUDA 实现（cuda.py:283）→ `get_valid_backends` → `validate_configuration`（backend.py:271，返回 list[str]）
- MLA: `MLAAttention`（mla_attention.py:323），DeepSeek-V2 提出的架构（KV cache 大幅缩小），先不深挖

**用户复述确认**:
- `validate_configuration` 返回 `list` 而非 `bool` = **一次收集全部失败原因**（启动一次看到所有不匹配项，而不是修一个报一个）✅
- `get_valid_backends` 判定"可用" = `validate_configuration` 返回空列表；ImportError（算子没装）走 except 分支也是失败原因之一 ✅
- `get_path` 返回的是模块路径字符串（目录+文件+类名）✅

**卡点**: 无（vLLM 代码链路已通到 backend 分发层；FA 内核细节待读 triton_unified_attention.py）

**下次入口**: 读 `vllm/v1/attention/ops/triton_unified_attention.py:505` 的 `@triton.jit unified_attention`，验证循环结构（外层 Q 块/内层 K 块）、m_i/l_i（online softmax）、tl.dot/acc（同步累积）——把 FA 理论接上 vLLM 内核

---

### 2026-08-05: FlashAttention 基本讨论（白天，与 AI 讨论）


**理解了什么**（用户复述 + 2026-08-05 晚间确认）:
- 标准 attention 的 IO 瓶颈：`S = QK^T` (N×N) 和 `P = softmax(S)` (N×N) 都要**写回 HBM 再读出来**——S 写/读 + P 写/读，共 4 次 O(N²) 的 HBM 往返
- FlashAttention 不写回中间的 scores/softmax：S/P 留在 SRAM，永远不碰 HBM
- 效果：HBM 访问量从 O(N²) 主导降下来（口语表述 O(N·D)；精确公式见下）

**疑问与解答**:

| Q | A |
|---|---|
| "访问量是 O(N·D)" 这个表述准确吗？ | 教学简化。严格论文公式：标准 attention HBM 访问 O(N²+ND)；FA 是 O(N²D²/M)，M=SRAM 大小。当 M≥D² 时远小于 O(N²)。FA 里每个 Q block 要扫所有 K/V blocks，K/V 被重复读，所以 N² 没完全消失，只是被 D²/M 因子压小 |
| S 和 P 两个矩阵产生几次 O(N²) 往返？ | 4 次：写 S、读 S（softmax 时）、写 P、读 P（乘 V 时） |
| Online softmax 的更新公式？ | `m = max(m_old, m_new)`；`l = l_old * e^(m_old-m) + l_new * e^(m_new-m)`。两半形式上都要乘 e^(局部max-全局max)，谁输谁被压缩（系数<1），赢家是 e^0=1。新 block 刷新全局 max 时 = 只补旧的（用户 2026-08-05 自己推导 ✅） |

**卡点**: 无

**下次入口**: 从 IO 复杂度 → Tiling/Online softmax 细节（如何滚动更新 softmax）

---

## 核心知识点（待验证）

> 此节内容必须在用户复述确认后才可填写。当前仅列提纲，未填充。

### 为什么需要 FlashAttention

- 标准 attention 的时间/显存复杂度：`O(N²)` 的 attention score 矩阵
- 问题：读 HBM 写 HBM 的 IO 瓶颈

### 核心思想

- Tiling（分块）：把 Q/K/V 切成 block，在 SRAM 里算
- Online softmax（重缩放 trick）：不存完整 score 矩阵，滚动更新 softmax
- 减少 HBM 读写：从 `O(N²)` 降到线性

### IO 复杂度（2026-08-05 已复述确认 ✅）

```
标准 attention  HBM 访问: O(N² + N·D)          ← N² 项来自 S/P 矩阵的 4 次往返
FlashAttention  HBM 访问: O(N²·D² / M)        ← M = SRAM 大小
当 M ≥ D² 时 → 远小于 O(N²)，接近线性

口语表述: Q/K/V/O 都是 (N×D)，只有它们需要 HBM 往返 → 看起来 O(N·D)
严格表述: 每个 Q block 要扫所有 K/V blocks，K/V 被重复读 → 真实 O(N²D²/M)
实践中: d=128, M=192KB → D²/M ≈ 0.085，比 O(N²) 压 10 倍以上
```

### Online Softmax（2026-08-05 用户自己推导 ✅）

```
滚动维护两个全局量: m = 见过的最大 score, l = 对齐到 m 的指数和

新 block 进来时:
  m_new = max(block 内 scores)
  m = max(m_old, m_new)
  l = l_old * e^(m_old - m)   ← 旧部分重缩放
    + l_new * e^(m_new - m)   ← 新部分重缩放（其中一个是 e^0 = 1）

精髓: 每次让所有已见分数对齐到当前全局 max，sum 永远正确 → 不存完整矩阵
谁输谁补: 新 block 刷新 max 时只补旧的（系数 e^(m_old-m_new)）；反之只补新的
```

### Tiling（分块，2026-08-05 用户自己推导 ✅）

**定义**：把大矩阵切成块，一块一块算，循环粒度是"块"不是"单个元素"。

**推导链**（用户从点积出发逐步推出）:
1. softmax 按**行**归一化（一行 = 一个 token 对全部 token 的权重）
2. 一个 query 要和**全部 N 个 key** 做点积（S 的一行有 N 个数）
3. 所以 Q 块要**遍历访问所有 K 块**才能凑齐一行

**循环结构**:
```
外层循环: 遍历 Q 块 (Q₀, Q₁, Q₂, Q₃ ...)
  内层循环: 遍历 K/V 块 (K₀, K₁, K₂, K₃ ...)
    算出一小块 scores (B_r × B_c) —— 留在 SRAM
    online softmax 更新 m/l
    【同时】用当前分数乘对应 V 块, 累积到 output ← 关键!
```

**两个关键设计**（用户推出）:
- 同步算 output：scores 算完立刻乘 V 累积，**不落地 HBM** → 这就是"无中间回写"的实现方式（也是 FA 叫 Flash 的原因）。只更新 m/l 不更新 O 就得存 scores，白做
- block_size 取舍：**往大选**（内层循环次数少、K/V 重读少、IO 省），但受 SRAM 容量约束（Q 块+K 块+V 块+O 块+中间量都要装下）

**block_size 数值推导**（2026-08-10 用户完整闭环 ✅）:

场景: A100, SRAM = 192KB = 196,608 bytes; head_dim d=64; fp16=2B, fp32=4B

```
一次内层循环的 SRAM 占用（B = block 行数）:
  Q 块  B×64 fp16 = 128B
  K 块  B×64 fp16 = 128B
  V 块  B×64 fp16 = 128B
  O 块  B×64 fp32 = 256B   ← 注意: O 用 fp32 累加器（Triton: acc = tl.zeros(..., tl.float32)）
  m     B 个     fp32 = 4B    ← softmax 按行 → 每行一个 max（不是标量）
  l     B 个     fp32 = 4B    ← 每行一个分母（不是 B×B 矩阵!）
  scores B×B     fp32 = 4B²
  固定部分合计: 648B，约束: 648B + 4B² ≤ 196,608

解:  B=155 → 196,540 ✓（差 68 bytes）; B=156 → 198,432 ✗
     → 纯 SRAM 数学极限 B ≈ 155（SRAM 用到 99.97%）
```

**为什么实际选 64/128 不选 155**（三个约束，用户逐步推出）:

| 约束 | 说明 |
|---|---|
| SRAM 留冗余 | 真实 kernel 还有量化 scale(k_scale/v_scale)、sink、block table、多 stage 流水产物 → SRAM 不能 100% 占满 |
| **寄存器（最硬约束）** | acc(B×d fp32) 在循环内反复读写 → **常住寄存器**。每线程寄存器=255: B=155 → acc 占 77/线程；B=64 → 32/线程。寄存器不够 → spill 到"本地内存"(物理=HBM) → 省掉的 HBM 访问全回来 → 性能暴跌 |
| 2 的幂对齐 | 64/128 是 2 的幂，硬件（线程调度/库函数/对齐）最友好；155 不是 |

**一句话结论**: block_size 不是选"SRAM 能装的最大值"，而是**同时满足 SRAM + 寄存器 + 2 的幂对齐三个约束的平衡点**。这也是 `BLOCK_M` 是 `tl.constexpr` 的原因（编译期决定 → 决定寄存器布局）。

**l 的命名**（用户困惑过）: l ≠ s ≠ acc。s 已被 scores 占用; acc 是 O 块累积器(B×d 矩阵); l 是**分母标量数组**(B 个)，online softmax 经典文献惯例 m/l 配对（m=max, l=分母）。

**点积基础**（seq=4, hidden=8 例子）:
```
q0 = [2, 0, 0, 0, 0, 0, 0, 0]   k0 = [1, 0, 0, 0, 0, 0, 0, 0]
点积 = 逐位相乘再求和: q0·k1 = 2×0 + 0×1 + ... = 0
S[i][j] = q_i · k_j → S 是 (4,4): 行数 = query 数, 列数 = key 数
S 第 0 行 = q0 对 k0,k1,k2,k3 的 4 个点积 → "一个 query 和 N 个 key 点积"
```

### 关键公式/图示

（待补）

---

## 相关代码

**vLLM 中 FA 的接入点**（2026-08-05 直接查源码确认，路径是新版 v1 结构）:

```
vllm/v1/attention/selector.py:53  get_attn_backend()     → backend 选择入口
vllm/v1/attention/backends/registry.py:44  FLASH_ATTN = "...flash_attn.FlashAttentionBackend"  → 注册表
vllm/v1/attention/backends/flash_attn.py:66  FlashAttentionBackend   → backend 外壳
vllm/v1/attention/backends/flash_attn.py:594  FlashAttentionImpl    → 实际实现
vllm/v1/attention/backends/flash_attn.py:682  forward()             → 调用 flash_attn_func 的地方
```

- backend 选择：`selector.get_attn_backend()` → `_cached_get_attn_backend` → `current_platform.get_attn_backend_cls(backend)` → 按平台选具体类（NVIDIA → FlashAttention，AMD → rocm_aiter_fa 等）
- `FlashAttentionImpl.forward`：query/key/value 形状 = [num_tokens, num_heads, head_size]，kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
- 版本检测：`get_flash_attn_version()`（FA2/FA3/FA4 自动选，head_size>256 且 SM90 强制 FA4）
- kv cache layout 由 backend 决定：`backend.get_required_kv_cache_layout()`（selector.py:131）

**⚠️ 重要架构事实**（2026-08-10 用户读代码发现）: FlashAttention backend 里**没有 Triton kernel**——它调用的是**外部库** dao-AILab/flash-attention 的 `flash_attn_varlen_func`（C++/CUDA，vLLM 之外）。vLLM 自己的 FA 内核在**另一个并行的 backend** 里:

```
FlashAttentionBackend (flash_attn.py)  → flash_attn_varlen_func  ← 外部库（默认）
TritonAttentionBackend (triton_attn.py) → unified_attention → kernel_unified_attention ← vLLM 自己写的（可选 --attention-backend triton）
```

**Triton kernel 与 FA 理论的对应**（2026-08-10 用户逐行验证 ✅）:

```
triton_attn.py:610        unified_attention(...)                     ← Python 包装
triton_unified_attention.py:505  def unified_attention(...)          ← 选 tile size / 处理量化
triton_unified_attention.py:58   @triton.jit kernel_unified_attention(...)  ← Triton 内核

内核内对应关系:
  170: offs_m = tl.arange(0, BLOCK_M)          ← 外层 Q 块（BLOCK_M = B）
  199: L = tl.full([BLOCK_M], 1.0, ...)        ← online softmax 的 l（每行一个标量 ✅）
  200: acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], tl.float32)  ← O 块 fp32 累加器 ✅
  299: S : (BLOCK_M, TILE_SIZE)                ← scores 块（TILE_SIZE = 内层 K 块）
  94:  BLOCK_SIZE (tl.constexpr)               ← 注意: 这是 KV cache 页大小(=16, PagedAttention),
                                                 只用于索引计算(seq_offset // BLOCK_SIZE), 不是循环块!

⚠️ 三个 "block" 的区别:
  BLOCK_M      = Q 块行数（循环粒度，= 学的 block_size B）
  TILE_SIZE    = 内层 K/V 块大小（循环粒度）
  BLOCK_SIZE   = KV cache 页大小（PagedAttention，索引用）← 最容易混淆
```

---

## 相关资源

- 原论文: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- （待补）

