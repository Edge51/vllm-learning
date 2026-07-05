# Session 1 回顾：从请求到模型执行

学习日期：2026-06-28

## 你理解了什么？

### 1. vLLM 的完整请求链路

```
请求进来
  → jinja 渲染（tools 格式化进 prompt）
  → tokenize（文本 → token IDs）
  → scheduler 调度
    → 新请求 → 分配尽量多的 token（prefill）
    → 已有请求 → 分配 1 个 token（decode）
  → model_runner 执行
    → 多个 token → prefill 路径（并行算 KV cache）
    → 1 个 token → decode 路径（复用 KV cache）
  → 返回结果
```

### 2. Prefill vs Decode
| | Prefill | Decode |
|---|---|---|
| 何时发生 | 请求第一次进入 | 已有 KV cache 后 |
| 计算量 | 大量（并行算所有 token 的 KV） | 小（只算 1 个新 token） |
| GPU 用途 | 矩阵乘法（计算密集型） | 显存读取（带宽密集型） |
| `num_scheduled_tokens` | > 1（通常几百） | = 1 |

### 3. KV Cache
- 缓存的是每层 attention 的 **K 和 V**（不是 Q）
- 一旦算好，后续 decode 直接读取，不用重算
- Q 每轮都变，不能缓存

### 4. 代码组织

```
请求入口
  vllm/entrypoints/openai/api_server.py

引擎主循环（每 step 调一次）
  vllm/v1/engine/core.py → step()
    │
调度器（决定每轮谁跑、跑多少 token）
  vllm/v1/core/sched/scheduler.py → schedule()
    │
模型执行器（实际调用模型 forward）
  vllm/v1/worker/gpu_model_runner.py → execute_model()
    │
Attention 后端（根据 query_len 隐式区分 prefill/decode）
  vllm/v1/attention/backends/flash_attn.py
```

### 5. Prefill/Decode 的"隐式传递"

scheduler 到 model_runner 之间没有 `if is_prefill` 这样的标签。

区分是靠 `num_scheduled_tokens` 这个数字隐式传递的：

```
scheduler:  request_A → num_scheduled_tokens = 512（"这是 prefill"）
            request_B → num_scheduled_tokens = 1  （"这是 decode"）
                  ↓
model_runner: max_query_len = max(512, 1)
              _is_uniform_decode(max_query_len) → False
              → 不走 decode 优化路径
                  ↓
attention 后端: query_len = 512 → 调 prefill kernel
                query_len = 1   → 调 decode kernel
```

## 你学会了什么技能

| 技能 | 下次用在哪 |
|---|---|
| `F12` 跳类型定义 | 任何 Python 项目 |
| 参数的值找调用方（搜调用点） | 任何带依赖注入的代码 |
| `type[Executor]` = 传类本身 | Python 工厂模式 |
| 动态字符串加载 `F12` 跳不动 → 搜 config 默认值 | 框架代码 |
| `git submodule` 切换版本分支 | ascend-vLLM 版本管理 |
