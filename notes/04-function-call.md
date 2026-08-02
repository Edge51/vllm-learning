# Phase 4：Function Call 全链路

> vLLM v0.20.2 + ascend-vllm (vllm-cloud-A5) Qwen3CoderToolParser
> 
> 学习日期: 2026-07-14 (Session 1: 主链路 + 流式分支)
>          2026-07-15 (Session 2: streaming required 算法 + 非流式分支 + delta filter + prev_tool_call_arr)
> 状态: 主线全链路已覆盖，核心算法（streaming required、非流式5分支、delta filter、共享状态）已完成

---

## 目录

1. [入口链路](#1-入口链路)
2. [Jinja 渲染链](#2-jinja-渲染链)
3. [adjust_request——请求参数调整](#3-adjust_request请求参数调整)
4. [engine_client.generate——发往引擎](#4-engine_clientgenerate发往引擎)
5. [Streaming vs 非流式输出处理](#5-streaming-vs-非流式输出处理)
6. [tool_parser 实现（以 Qwen3CoderToolParser 为例）](#6-tool_parser-实现以-qwen3codertoolparser-为例)
7. [JSON Schema 构建（guided decoding）](#7-json-schema-构建guided-decoding)
8. [投机解码对 tool parser 的影响](#8-投机解码对-tool-parser-的影响)
9. [影响 function call 的特性清单](#9-影响-function-call-的特性清单)
10. [代码中的架构设计讨论](#10-代码中的架构设计讨论)
11. [extract_tool_call_required_streaming——required/named 路径](#11-extract_tool_call_required_streamingrequirednamed-路径)
12. [_parse_tool_calls_from_content——非流式工具调用解析](#12-_parse_tool_calls_from_content非流式工具调用解析)
13. [辅助函数：_filter_delta_text / make_tool_call_id / finish_reason](#13-辅助函数_filter_delta_text--make_tool_call_id--finish_reason)
14. [共享状态 prev_tool_call_arr / streamed_args_for_tool](#14-共享状态-prev_tool_call_arr--streamed_args_for_tool)

---

## 1. 入口链路

```
vllm serve <model>
  └→ cli/main.py:main() → cli/serve.py:ServeSubcommand.cmd()
      └→ api_server.py:run_server() → setup_server()
          └→ 初始化 EngineClient (AsyncLLM)
          └→ 初始化 OpenAIServingRender
          └→ 初始化 OpenAIServingChat
          └→ 注册路由 /v1/chat/completions
```

```
create_chat_completion()                          # chat_completion/serving.py:229
  ├→ render_chat_request(request)                 # line 251 → serve/render/serving.py
  │   └→ render_chat(request)                     # line 184
  │       └→ preprocess_chat()                    # line 523
  │           ├→ renderer.render_chat_async()     # BaseRenderer → HfRenderer
  │           ├→ reasoning_parser.adjust_request()
  │           └→ tool_parser.adjust_request()
  │
  ├→ engine_client.generate(engine_input, ...)    # line 341 → AsyncLLM
  │
  ├─ [stream=True]
  │   └→ chat_completion_stream_generator()       # line 363
  │
  └─ [stream=False]
      └→ chat_completion_full_generator()          # line 375
```

---

## 2. Jinja 渲染链

### 调用链

```
render_chat()
  └→ preprocess_chat()
      └→ renderer.render_chat_async()             # base.py:1005
          └→ render_messages_async()              # hf.py:681
              └→ safe_apply_chat_template()        # hf.py:460
                  ├→ resolve_chat_template()       # 4 优先级找模板字符串
                  ├→ resolve_chat_template_kwargs()# 解析 jinja 模板变量
                  └→ tokenizer.apply_chat_template() # HuggingFace → 实际 jinja 渲染
```

### 模板字符串的加载

```python
# api_server.py:352
resolved_chat_template = load_chat_template(args.chat_template)
#                    → cached_lru_cache(_load_chat_template)
#                    → 读文件 / 内置模板 / 字面量
```

`load_chat_template` 只返回纯字符串。**真正编译和执行 jinja 模板发生在 HuggingFace 的 `apply_chat_template` 里。**

### vLLM 里直接使用 jinja2 的地方

只有 `hf.py:372`：

```python
def _resolve_chat_template_kwargs(chat_template: str) -> Set[str]:
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True,
        extensions=[AssistantTracker, jinja2.ext.loopcontrols],
    )
    parsed_content = env.parse(chat_template)            # 编译 AST
    template_vars = jinja2.meta.find_undeclared_variables(parsed_content)
    return template_vars                                 # 找出模板用了哪些变量
```

自定义 Extension：`AssistantTracker`（`hf.py:361`），支持 `{% generation %}...{% endgeneration %}` 标签。

### 不走 jinja 的模型

| 模型族 | 渲染方式 | 原因 |
|---|---|---|
| Qwen/LLaMA/GLM 等 | `HfTokenizer.apply_chat_template()` → jinja | 标准 HuggingFace 路径 |
| DeepSeek V4 | `encode_messages()` 自定义编码 | 自研 tokenizer，覆写 `apply_chat_template` |
| GPT-OSS | `_make_request_with_harmony()` | 使用 Harmony 格式，不走 jinja |
| Mistral | `MistralRenderer` | 使用 `mistral_common` 自研编码 |

---

## 3. adjust_request——请求参数调整

**作用：在 prompt 渲染之前，修改请求参数。不是后处理。**

### tool_parser.adjust_request()

```python
# abstract_tool_parser.py:85
def adjust_request(self, request):
    json_schema = get_json_schema_from_tools(tool_choice, tools)
    if json_schema is not None:
        request.structured_outputs = StructuredOutputsParams(json=json_schema)
    return request
```

当 `tool_choice="required"` 或 `ChatCompletionNamedToolChoiceParam` 时，构建 JSON Schema 约束模型输出。

各模型子类在此基础上加定制：

| 模型 | 定制 |
|---|---|
| Hermes | `request.skip_special_tokens = False` |
| GLM-4-MoE | `required`/`named` 时不调 `super()`（GLM 输出 XML 不是 JSON） |
| DeepSeek V3.2 | `request.skip_special_tokens = False` |

### reasoning_parser.adjust_request()

```python
# abs_reasoning_parsers.py:171（默认空实现）
# gemma4_reasoning_parser.py:59 示例
def adjust_request(self, request):
    request.skip_special_tokens = False  # 保留 boundary tokens
    return request
```

### 调用顺序

```python
# preprocess_chat() 里
request = reasoning_parser.adjust_request(request)   # 先改 reasoning 参数
request = tool_parser.adjust_request(request)         # 再加工具约束
```

---

## 4. engine_client.generate——发往引擎

`engine_client` 的实际类型是 **`AsyncLLM`**（`v1/engine/async_llm.py`），继承自 `EngineClient`（`engine/protocol.py`）。

```python
# async_llm.py:524
async def generate(self, prompt, sampling_params, request_id, ...):
    q = await self.add_request(request_id, prompt, params, ...)
    # output_handler 后台 task 从 EngineCore 拉结果塞进队列
    while not finished:
        out = q.get_nowait() or await q.get()   # 从 per-request 队列拿
        finished = out.finished
        yield out
```

### add_request 内部

```python
async def add_request(self, ...):
    # 1. input_processor.process_inputs(prompt)  → EngineCoreRequest
    request = self.input_processor.process_inputs(request_id, prompt, params, ...)

    # 2. output_processor.add_request(request)   → 登记 output 处理
    self.output_processor.add_request(request, ...)

    # 3. engine_core.add_request_async(request)  → 发送到 EngineCore 进程
    await self.engine_core.add_request_async(request)
```

### input_processor.process_inputs 做什么

**不是 encoder 输入处理。是把 `EngineInput`（renderer 产出）转成 `EngineCoreRequest`（带调度信息的请求）。**

```python
def process_inputs(self, request_id, prompt, params, ...) -> EngineCoreRequest:
    # 1. 校验参数
    # 2. 如果还没 tokenize，走 input_preprocessor.preprocess()
    # 3. 分离 encoder/decoder 输入（只为 T5 这类 encoder-decoder 模型）
    encoder_inputs, decoder_inputs = split_enc_dec_input(processed_inputs)
    # 4. 处理 multimodal 特征（图片、视频、音频）
    if decoder_inputs["type"] == "multimodal":
        # 提取 mm_features
        ...
    # 5. 组装 EngineCoreRequest
    return EngineCoreRequest(request_id=..., prompt_token_ids=..., mm_features=..., ...)
```

### 进程架构

```
AsyncLLM (API server 进程)
  ├─ engine_core = EngineCoreClient (IPC 代理)  →   EngineCoreProc (独立进程)
  │                                                    ├─ Scheduler
  │                                                    └─ Model Runner
  └─ output_processor (独立进程内)
       └─ 后台 output_handler loop
            ├─ engine_core.get_outputs_async()
            └─ output_processor.process_outputs() → queue
```

---

## 5. Streaming vs 非流式输出处理

### 非流式

```python
# chat_completion_full_generator():1278
async for res in result_generator:
    # 等待完整输出
    ...

# 收集完整文本后
tool_call_info = tool_parser.extract_tool_calls(model_output, request)
# 一行提取，不需要状态
```

### 流式

```python
# chat_completion_stream_generator():530
async for res in result_generator:          # 每个 step 迭代一次
    delta_text = output.text                # 本轮新增文本
    current_text = previous_text + delta_text

    # 按模型路径分流
    if is_mistral_grammar_path:
        result = tool_parser.extract_maybe_reasoning_and_tool_streaming(
            previous_text, current_text, delta_text,
            previous_token_ids, current_token_ids, output_token_ids, ...)
    elif tool_choice_auto:
        result = tool_parser.extract_tool_calls_streaming(
            previous_text, current_text)
    ...
```

`delta_text`：每个 step 的增量文本（可能对应 1 个或多个 token，取决于投机解码）。

### `_filter_delta_text`

```python
# serving.py:405 — 只用于 JSON 格式的 required tool_choice
def _filter_delta_text(delta_text, previous_text):
    bracket_level = _bracket_level(previous_text)
    # 逐字符扫描，只在 bracket_level != 0 时输出
    # 到达 level 0 时停止（工具定义结束）
```

---

## 6. tool_parser 实现（以 Qwen3CoderToolParser 为例）

> 文件：`ascend_vllm/tool_parsers/qwen35_tool_parser.py`

### 模型输出格式（XML）

```xml
<tool_call>
  <function=get_weather>
    <parameter=location>北京</parameter>
    <parameter=unit>celsius</parameter>
  </function>
</tool_call>
```

### Sentinel tokens

```python
self.tool_call_start_token = "<tool_call>"
self.tool_call_end_token   = "</tool_call>"
self.tool_call_prefix      = "<function="
self.function_end_token    = "</function>"
self.parameter_prefix      = "<parameter="
self.parameter_end_token   = "</parameter>"
```

### 非流式 `extract_tool_calls`

```python
def extract_tool_calls(self, model_output, request):
    # 1. 快速过滤
    if self.tool_call_prefix not in model_output:
        return ExtractedToolCallInformation(tools_called=False, content=model_output)

    # 2. 正则找 <tool_call>...</tool_call>
    function_calls = self._get_function_calls(model_output)

    # 3. 逐个解析 XML → ToolCall
    tool_calls = [self._parse_xml_function_call(fc, request.tools) for fc in function_calls]

    # 4. 提取 tool_call 前面的内容
    content = model_output[:找到第一个 <tool_call> 的位置]

    return ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls, content=content)
```

`_parse_xml_function_call` 内部：

```python
def _parse_xml_function_call(self, function_call_str, tools):
    # 从 "<function=name>..." 提取函数名
    function_name = function_call_str[:第一个 ">"]

    # 用正则 <parameter=name>value</parameter> 提取参数
    for match in self.tool_call_parameter_regex.findall(parameters):
        param_dict[param_name] = self._convert_param_value(...)

    return ToolCall(function=FunctionCall(name=function_name, arguments=json.dumps(param_dict)))
```

### 流式 `extract_tool_calls_streaming`——状态机

状态变量：

```python
is_tool_call_started: bool      # 是否已开始 tool call
in_function: bool               # 是否在 <function=...> 内部
in_param: bool                  # 是否在 <parameter=...> 内部
json_started / json_closed: bool # 参数 JSON 构建状态
current_tool_index: int         # 当前处理第几个 tool call
header_sent: bool               # 函数名 header 是否已发
param_count: int                # 当前函数已处理完的参数个数
accumulated_params: dict        # 累积的参数名→值
streamed_args_for_tool: list    # 每个 tool call 已发送的 JSON 片段
```

每次收到新 token：

```python
def extract_tool_calls_streaming(self, previous_text, current_text, delta_text, ...):
    # 首次调用 → 重置状态
    if not previous_text:
        self._reset_streaming_state()

    # delta_text 为空 → EOS 处理
    if not delta_text:
        # 检查 tool call 是否完整，返回空 content 或 None
        ...

    self.accumulated_text = current_text

    # 状态机主循环
    if not self.is_tool_call_started:
        # 检测 <tool_call> 开始
        if self.tool_call_start_token in delta_text:
            self.is_tool_call_started = True

    if self.is_tool_call_started and not self.in_function:
        # 检测 <function=name> → 发送 DeltaToolCall(name=name)
        ...

    if self.in_function:
        # 处理参数: <parameter=name>value</parameter>
        # (loop 处理所有完整参数 → JSON fragments)
        ...

    # 检测 </function> → 关闭参数 JSON
    # 检测 </tool_call> → 整轮完成，重置状态
```

---

## 7. JSON Schema 构建（guided decoding）

```python
# tool_parsers/utils.py:172
def _get_tool_schema_from_tool(tool):
    name, params = _extract_tool_info(tool)    # ← tool.name, tool.parameters (JSON Schema)
    return {
        "properties": {
            "name": {"type": "string", "enum": [name]},
            "parameters": params,              # ← 用户定义的参数 schema
        },
        "required": ["name", "parameters"],
    }

# tool_parsers/utils.py:203
def _get_json_schema_from_tools(tools):
    return {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "anyOf": [_get_tool_schema_from_tool(t) for t in tools],
        },
    }
```

`adjust_request` 只对 `tool_choice="required"` 或 `ChatCompletionNamedToolChoiceParam` 设 `structured_outputs`。`tool_choice="auto"` 时不约束格式。

| `tool_choice` | guided decoding | 原因 |
|---|---|---|
| `"none"` | ❌ | 无工具调用 |
| `"auto"` | ❌ | 模型自由选择 |
| `"required"` | ✅ | 必须输出工具调用，需要 JSON 约束 |
| 指定某函数 | ✅ | 必须输出指定格式 |

---

## 8. 投机解码对 tool parser 的影响

### 根因

投机解码在一个 step 内接受多个 draft token → `delta_text` 一次包含多个 XML 标记。

```
正常解码:
  delta_text: "<" → "f" → "u" → "n" → "c" → ... → ">"
  状态机逐字符推进

投机解码（接受 5 个 draft token）:
  delta_text: "<function=get_weather>"
  状态机直接从 False 跳到 "完整标签"
```

### Qwen35 parser 里已有的投机修复

| 位置 | 问题 | 修复 |
|---|---|---|
| L461 | 一个 delta 同时包含 `{` 和参数数据 | 强制先发 `{`，不管参数前缀 |
| L502-508 | 一个 delta 包含多个完整参数 | `while` 循环处理**所有**已完成参数 |
| L603-607 | 参数值 + `</function>` 一起到 | 先处理参数循环，再检查 `</function>` |

### 为什么还存在问题

状态机依赖多步 `if/else` 链推进。投机场景下一个 `delta_text` 可能跨越多个状态（`is_tool_call_started` → `in_function` → `param_count++`），但代码里仍有依赖单步推进的路径没覆盖到。

### 推荐修复策略：无状态解析 + diff

```python
def extract_tool_calls_streaming(self, previous_text, current_text, delta_text, ...):
    # 阶段 1: 无状态完整解析 current_text
    full_state = self._parse_full_text(current_text)

    # 阶段 2: 对比缓存的上次状态 → 计算 delta
    delta = self._diff_state(self._cached_state, full_state)

    # 阶段 3: 更新缓存
    self._cached_state = full_state

    return self._delta_to_message(delta)
```

`_parse_full_text` 是纯函数，不依赖状态变量。`delta_text` 一次来几个 token 都不影响正确性。

JSON 格式（Hermes 等）用 `partial_json_loads` 天然无状态，不受投机影响。

---

## 9. 影响 function call 的特性清单

| 特性 | 影响等级 | 说明 |
|---|---|---|
| **Speculative decoding** | 🔴 高 | `delta_text` 批量化导致状态机不同步。XML 类 parser 尤其脆弱 |
| **Chunked prefill** | 🟡 中 | prefill 阶段 `delta_text=""` 但可能有 token_ids，parser 需正确处理空 delta |
| **并行 tool call（单请求多调用）** | 🔴 高 | `current_tool_index`、`header_sent`、参数边界管理。单个请求内连续多个 `<tool_call>` |
| **Reasoning + tool 混合** | 🔴 高 | 模型先输出 reasoning 再输出 tool call。两个 parser 共享 delta_message 流 |
| **n > 1（multiple choices）** | 🟡 中 | 每个 choice 独立维护 parser 状态，`parsers[i]` 数组管理 |
| **Beam search** | 🟢 低 | beam search + tool call 目前支持不完整 |
| **Logprobs** | 🟢 低 | tool call token 的 logprob 计算方式 |
| **Grammar / Guided decoding** | 🔴 高 | grammar 约束输出格式，和 tool parser 解析逻辑必须一致 |
| **EOS token 处理** | 🟡 中 | 模型可能在 tool call 中间输出 EOS，需要截断处理 |
| **Streaming 断连重连** | 🟢 低 | parser 状态需要能重新初始化 |
| **Multi-modal（图片/视频/音频）** | 🟡 中 | 多模态 token 穿插在 tool call 中，影响 delta_text 分割 |

---

## 10. 代码中的架构设计讨论

### Mistral / Harmony 的 `if/else` 分支

Mistral 和 GPT-OSS（`use_harmony`）的代码嵌入在主线流程中，原因是：

- `mistral_common` 不是 HuggingFace tokenizer，有自研的 grammar 约束和 prompt 编码
- 引入时没有重构插件架构，走了 `if/else` 最小改动路径
- 目前没有第三个模型族进来，重构动力不足

影响：`chat_completion_stream_generator` 里约 15 处 `is_mistral_*` 分支，代码可读性差。

更好的设计：入口分叉 → 两个独立的 pipeline（prompt 编码 + output 提取）→ 公用 SSE 组装。

### DPEngineCore / dp_rank

不是 P/D 分离（disaggregation），是 **Expert Parallelism（EP）**。

```python
data_parallel_size    # MoE 专家分片数
data_parallel_rank    # 当前进程的专家分片编号
dp_rank               # = 我的 expert 分片编号
DPEngineCoreProc      # 支持跨 rank all-to-all 通信的 EngineCore
```

### MoE（Mixture of Experts）

```python
class MoE(nn.Module):
    def forward(self, x):
        weights, indices = self.router(x)        # 每个 token 选 top-2 专家
        return fused_moe(x, self.experts, indices, weights)
```

- 总参数量大（如 DeepSeek V3: 256 × ~5.5B = 1.4T）
- 推理时只激活 2 个专家，计算量 ≈ 2× 普通模型
- 专家可以分到多张 GPU（vLLM 的 DP/EP 做这个）

### 关于 `lru_cache` 装饰器

```python
# 等价于 @lru_cache
_cached_load_chat_template = lru_cache(_load_chat_template)
```

装饰器 = 接受函数作为参数、返回新函数的函数。`lru_cache` 包装后，相同参数的调用直接返回缓存结果，不重复执行函数体。

`lru_cache` 支持两种用法（duck typing）：

```python
@lru_cache                  # maxsize 被赋值为被装饰的函数 → callable 分支
@lru_cache(maxsize=256)     # maxsize=256 → int 分支
```

---

### 相关文件索引

```
# 入口与服务
third_party/vllm/vllm/entrypoints/openai/chat_completion/serving.py
  └─ create_chat_completion()        (line 229)
  └─ chat_completion_stream_generator() (line 530)
  └─ chat_completion_full_generator()   (line 1278)
  └─ extract_tool_call_required_streaming() (line 430)
  └─ _filter_delta_text()             (line 405)

# 渲染
third_party/vllm/vllm/entrypoints/serve/render/serving.py
  └─ OpenAIServingRender
      └─ render_chat_request()        (line 120)
      └─ render_chat()                (line 184)
      └─ preprocess_chat()            (line 523)
      └─ _make_request_with_harmony() (line 395)

third_party/vllm/vllm/renderers/hf.py
  └─ HfRenderer
      └─ render_messages_async()      (line 681)
      └─ safe_apply_chat_template()   (line 460)
      └─ resolve_chat_template()      (line 96)
      └─ resolve_chat_template_kwargs() (line 407)
      └─ _resolve_chat_template_kwargs() (line 372)
      └─ AssistantTracker             (line 361)

# 引擎
third_party/vllm/vllm/engine/protocol.py
  └─ EngineClient (ABC)               (line 40)

third_party/vllm/vllm/v1/engine/async_llm.py
  └─ AsyncLLM                         (line 70)
  └─ generate()                       (line 524)
  └─ add_request()                    (line 280)

third_party/vllm/vllm/v1/engine/input_processor.py
  └─ InputProcessor                   (line 36)
  └─ process_inputs()                 (line 234)

# Tool Parsers（vLLM 主线）
third_party/vllm/vllm/tool_parsers/abstract_tool_parser.py
  └─ ToolParser (ABC)
  └─ adjust_request()                 (line 85)
  └─ extract_tool_calls()             (line 128)
  └─ extract_tool_calls_streaming()   (line 138)

third_party/vllm/vllm/tool_parsers/utils.py
  └─ _get_tool_schema_from_tool()     (line 172)
  └─ _get_json_schema_from_tools()    (line 203)
  └─ get_json_schema_from_tools()     (line 220)
  └─ _extract_tool_info()             (line 147)

# Tool Parsers（ascend-vllm）
third_party/vllm-cloud-main/ascend_vllm/tool_parsers/qwen35_tool_parser.py
  └─ Qwen3CoderToolParser
      └─ extract_tool_calls()         (line 259)
      └─ extract_tool_calls_streaming() (line 303)
      └─ _parse_xml_function_call()   (line 217)
      └─ _get_function_calls()        (line 243)

# Config
third_party/vllm/vllm/config/parallel.py
  └─ ParallelConfig (data_parallel_*) (line 108)
```

---

---

## 11. extract_tool_call_required_streaming——required/named 路径

> 文件: `vllm/entrypoints/openai/chat_completion/serving.py` (line ~430)
> 触发条件: `tool_choice == "required"` 或 `tool_choice` 为 `NamedToolChoice`

### 11.1 函数签名与地位

```python
def extract_tool_call_required_streaming(
    self,
    all_outputs: list[ChatCompletionStreamOutput],  # 当前步所有 output idx
    request: ChatCompletionRequest,                  # 原始请求
    merge_delta: bool = False,                       # 是否合并到前一个 tool_call
    prev_tool_calls: list[DeltaToolCall] | None = None,  # flow-ctl 传入的之前 tool_call
) -> extrat_tool_call_required_streaming_ReturnType:
```

`extrat_tool_call_required_streaming_ReturnType` 是一个 NamedTuple:

```python
class extrat_tool_call_required_streaming_ReturnType(NamedTuple):
    tool_calls: list[DeltaToolCall]  # 本轮生成的 tool call delta
    is_continue: bool                 # 是否在同一个 tool call 中间（未结束）
    prev_tool_call_arr: list[DeltaToolCall]  # 之前所有未完成的 tool_call（供下次流入）
    streamed_args_for_tool: list[str]        # 每个 tool_call 累积的 arguments 文本
```

### 11.2 总体流程

```
extract_tool_call_required_streaming()
  │
  ├─ [prev_tool_calls 非空] → prev_tool_call_arr = prev_tool_calls
  └─ [prev_tool_calls 为空] → prev_tool_call_arr = 从缓存 + 本轮 outputs 重建
  │
  ├─ [streamed_args_for_tool 缓存为空] → 初始化为 ["", "", ...]
  │
  ├─ 遍历 each_output in all_outputs:
  │   ├─ _filter_delta_text(output)        ← bracket level 过滤
  │   ├─ output.token_ids 累加到 token_buffer
  │   ├─ 用 partial_json_loads 尝试解析 JSON
  │   ├─ [存在前一个未完成的 tool_call]:
  │   │   └─ finishes_previous_tool() 判断
  │   │       ├─ [True] → 修复 spec-decode 导致的分片，追加到前一个
  │   │       └─ [False] → 正常追加
  │   ├─ [function_name_returned]:
  │   │   └─ True  → 后续条目: delta_text 直接透传, name=None
  │   │   └─ False → 首个条目: 用 regex 从 current_text 提取 name
  │   │              调用 make_tool_call_id() 生成 id
  │   └─ 组装 DeltaToolCall {id, type, function={name, arguments}}
  │
  └─ return (tool_calls, is_continue, prev_tool_call_arr, streamed_args_for_tool)
```

### 11.3 关键机制

**`partial_json_loads`** (`tool_parsers/utils.py:123`):
- 对不完整的 JSON 文本进行容忍解析
- 只提取"到目前为止已经完整"的部分
- 配合 `_filter_delta_text` 先过滤掉末尾不完整的 token，再传给 parser

**`finishes_previous_tool`** (内联逻辑):
- 用 `obj[-2]` 取倒数第二个工具调用的 arguments（投机解码可能跨越两个 tool call）
- 如果 `partial_json_loads` 返回的倒数第二个 tool call 的 arguments 和当前 `streamed_args_for_tool` 不匹配 → 说明 spec-decode 把属于前一个 tool call 的 token 带到了当前步 → 回滚追加

**`function_name_returned` 分支**:
- `False`（首个条目）: 用 regex 从 `current_text` 提取 function name → 生成 `id` + `name`
- `True`（后续条目）: `delta_text` 直接设为 `arguments`, `name=None`

**`is_continue` 判断**:
- 基于 `prev_tool_call_arr` 是否还有未完成的输出
- 如果还有 tool_call 没 finishes → `is_continue=True`（继续在当前 tool_call 上追加）

---

## 12. _parse_tool_calls_from_content——非流式工具调用解析

> 文件: `vllm/entrypoints/openai/engine/serving.py` (line ~607)
> 触发: 非流式输出 `chat_completion_full_generator()` 中，拿到完整输出文本后调用

### 12.1 函数签名

```python
def _parse_tool_calls_from_content(
    self,
    request: ChatCompletionRequest,
    assistant_content: str,       # 模型生成的完整输出文本
    finish_reason: str | None,   # 来自 engine 的 finish_reason
) -> list[DeltaToolCall] | None:
```

返回 `None` 表示"这不是工具调用"（自然语言回复）。

### 12.2 5 分支决策树

```
_parse_tool_calls_from_content(request, assistant_content, finish_reason)
  │
  ├─ [branch 1] finish_reason 不是 "tool_calls" → return None
  │
  ├─ [branch 2] tool_choice == "required" 且 tool_parser 存在 →
  │   └─ tool_parser.extract_tool_calls()  ← 委托给 parser（XML 等自定义格式）
  │
  ├─ [branch 3] tool_choice == "required" 且 tool_parser 不存在 →
  │   └─ parse_tool_call_json(assistant_content)  ← 用 finish_reason=tool_calls 判断
  │       └─ 从 content 解析 JSON
  │
  ├─ [branch 4] tool_choice == "auto" / "named" 且 tool_parser 存在 →
  │   └─ tool_parser.extract_tool_calls()  ← 委托给 parser
  │
  └─ [branch 5] tool_choice == "auto" / "named" 且 tool_parser 不存在 →
      └─ load_tool_call_ids_and_delta_texts(request, finish_reason)
          与 load_tool_call_deltas(request, assistant_content, finish_reason, ...)
      └─ 基于 finish_reason 决定是否调用 parse_tool_call_json()
```

### 12.3 分支说明

| 分支 | tool_choice | 有 parser? | finish_reason | 行为 |
|------|------------|-----------|---------------|------|
| 1 | 任意 | — | 非 "tool_calls" | 直接 return None |
| 2 | required | ✅ | "tool_calls" | 走 parser.extract_tool_calls() |
| 3 | required | ❌ | "tool_calls" | 走 parse_tool_call_json() |
| 4 | auto/named | ✅ | "tool_calls" | 走 parser.extract_tool_calls() |
| 5 | auto/named | ❌ | "tool_calls" | 走 load_tool_call_deltas() + parse_tool_call_json() |

**核心原则**: 有 parser 就交给 parser，没有 parser 就用通用 JSON 解析。

---

## 13. 辅助函数：_filter_delta_text / make_tool_call_id / finish_reason

### 13.1 _filter_delta_text

> `serving.py:405-428` — 7 行核心 + 3 行边界

```python
@staticmethod
def _filter_delta_text(delta_text: str) -> tuple[str, list[int]]:
```

**作用**: 过滤掉 delta_text 末尾不完整的 JSON fragment。因为 speculative decoding 会把多个 token 的 delta 合并传入，末尾可能被截断。

**算法**:
```
bracket_level = 0
result = []
truncated_ids = []

逐字符扫描:
  '{' → bracket_level += 1
  '}' → bracket_level -= 1
  非 bracket → 追加到 result

扫描结束，bracket_level 可能:
  = 0: 完整 JSON
  > 0: 末尾缺少 } → 把从最后一个不匹配 { 开始的内容截掉
  < 0: 末尾缺少 { → 不可能，filter 最少返回 "{}"
```

**`_bracket_level`**: 所有状态都基于括号层级，不关心 JSON 语义。

### 13.2 make_tool_call_id

> `chat_utils.py:1706-1711` — 仅 6 行

```python
def make_tool_call_id(
    tool_call_id_type: str,
    history_tool_call_cnt: int,
    tool_name: str | None = None,
) -> str:
```

**两种格式**:

| format | 条件 | 示例 |
|--------|------|------|
| `call_<uuid4 16hex>` | 默认 | `call_a1b2c3d4e5f6a7b8` |
| `functions.<name>:<idx>` | Qwen/Kimi K2 风格 | `functions.get_weather:0` |

- `tool_call_id_type` 参数由 `request.tool_call_id_type` 控制
- `history_tool_call_cnt` 是跨请求累积的计数器，确保 id 全局唯一（应对多次 tool call 返回的情况）
- 单次请求内并行 tool call 的编号由 parser 自行处理

### 13.3 finish_reason 逻辑

> 在 `chat_completion_stream_generator` (line ~1170-1179) 和 `chat_completion_full_generator` (line ~1680) 中

**流式场景**:
```python
if finish_reason == "tool_calls":
    if tool_choice_auto:
        finish_reason = "tool_calls"   # auto 模式 → 透传
    elif tool_choice == "required":
        finish_reason = "tool_calls"   # required 模式 → 透传
    elif tool_choice_function_name:
        finish_reason = "stop"         # named 模式 → 转为 "stop"
```

**非流式场景**: `finish_reason` 由 engine 返回的 `output.finish_reason` 决定，在 `_parse_tool_calls_from_content` 中作为分支判断条件之一。

**关键逻辑**: `tool_choice={function:{name:"get_weather"}}`（named）时，即使 engine 返回 `tool_calls`，对客户端也要显示为 `"stop"`（用户只看得到自然语言回复，工具调用对用户透明）。

---

### 🗣 费曼检查点

> 讲清楚：
> 1. 一个 function call 请求从 HTTP 进来，经过哪些步骤才送到模型？
> 2. jinja 模板在什么时候、被谁渲染？渲染结果是什么？
> 3. `adjust_request` 是预处理还是后处理？它改了 request 的哪些字段？
> 4. 流式场景下 tool parser 怎么知道一个 tool call 已经完整了？
> 5. 投机解码为什么会导致 tool parser 解析错误？XML 格式和 JSON 格式哪个更脆弱？
> 6. `extract_tool_call_required_streaming` 中 `partial_json_loads` 和 `_filter_delta_text` 怎么配合？
> 7. `_parse_tool_calls_from_content` 的 5 个分支分别对应什么场景？
> 8. `make_tool_call_id` 的两种格式分别在哪用？`history_tool_call_cnt` 解决了什么问题？
> 9. `prev_tool_call_arr` 和 `streamed_args_for_tool` 分别存什么、谁写的、谁读的？二者之差用来干什么？

---

## 14. 共享状态 prev_tool_call_arr / streamed_args_for_tool

> 定义: `abstract_tool_parser.py:63,67`（基类），各 parser override
> Qwen3Coder: `qwen35_tool_parser.py:38,41`

这两个变量是所有 streaming 逻辑之间的**共享状态桥梁**，连接 parser、serving 层、finish_reason 补丁。

### 14.1 定义

```python
# abstract_tool_parser.py
self.prev_tool_call_arr: list[dict] = []       # parser 的"预期最终值"
self.streamed_args_for_tool: list[str] = []     # 已实际 stream 给客户端的内容
```

**`prev_tool_call_arr`**: 每个元素是一个 dict `{"name": "...", "arguments": "..."}`，arguments 是 JSON 字符串。这个数组存的是 parser 从模型输出中解析出的"理论最终结果"。

**`streamed_args_for_tool`**: 每个元素是截至当前步已经以 DeltaMessage 形式发给客户端的 arguments 文本。

两者按 tool call index 对齐：
```
prev_tool_call_arr[0] = {"name": "get_weather", "arguments": '{"city": "Beijing"}'}
streamed_args_for_tool[0] = '{"city": "Beij'  # 还没发完
```

### 14.2 Qwen3Coder 中四个关键站点

每个站点同步更新这两个变量，但视角不同：

#### 站点 ①：非流式 extract_tool_calls()（qwen35_tool_parser.py:276-285）

非流式场景一次拿到完整 XML 文本，解析后写入 prev_tool_call_arr：

```python
self.prev_tool_call_arr.clear()
for tool_call in tool_calls:
    self.prev_tool_call_arr.append({
        "name": tool_call.function.name,
        "arguments": tool_call.function.arguments,  # JSON 字符串
    })
```

只写 prev_tool_call_arr（非流式不需要 tracking 进度）。

#### 站点 ②：流式发现函数名（qwen35_tool_parser.py:431-443）

XML 状态机走到 `<function=xxx>` 时，两个数组同时初始化：

```python
self.prev_tool_call_arr.append({
    "name": self.current_function_name,
    "arguments": "{}",           # 占位，后续逐参数填充
})
self.streamed_args_for_tool.append("")  # 初始空字符串
```

此时 `streamed_args_for_tool[len-1] = ""` 表示这个 tool call 还没发出任何 arguments delta。

#### 站点 ③：逐参数追加（qwen35_tool_parser.py:584-585）

XML 状态机解析到一个 `<parameter=k>v</parameter>` 时，序列化成 JSON fragment 后追加到 streamed_args_for_tool，同时作为 delta 发出：

```python
self.streamed_args_for_tool[self.current_tool_index] += combined
```

streamed_args_for_tool 在这里是**增量累积**——每步追加，与 delta_text 同步。

#### 站点 ④：\</function\> 闭合（qwen35_tool_parser.py:620-632）

XML 结束时，用完整解析结果更新 prev_tool_call_arr（把占位的 "{}" 替换为完整 arguments）：

```python
self.prev_tool_call_arr[self.current_tool_index]["arguments"] = parsed_tool.function.arguments
self.streamed_args_for_tool[self.current_tool_index] += "}"
```

prev_tool_call_arr 在这里被**修正为最终值**（从 "{}" 变成完整 JSON）。

### 14.3 serving 层如何消费

`_should_check_for_unstreamed_tool_arg_tokens` 的补丁逻辑（serving.py:1119-1163）：

```
finish_reason 到 → output.finish_reason is not None
  │
  ├─ auto_tools_called = len(tool_parser.prev_tool_call_arr) > 0
  │   如果 parser 解析出了 tool call → prev_tool_call_arr 非空
  │
  ├─ expected_call = prev_tool_call_arr[index]["arguments"]
  │   = parser 认为模型应该生成的完整 arguments JSON
  │
  ├─ actual_call = streamed_args_for_tool[index]
  │   = 截至当前步已 stream 到客户端的 arguments 文本
  │
  ├─ 减掉最后一帧 delta 长度:
  │   actual_call = actual_call[:-latest_delta_len]
  │
  └─ remaining_call = expected_call.replace(actual_call, "", 1)
       stream 缺失部分 → 构造 final delta 补发
```

### 14.4 为什么需要两个变量

| 变量 | 谁写 | 谁读 | 用途 |
|------|------|------|------|
| `prev_tool_call_arr` | parser 的 XML 状态机（站点②④）或非流式解析（站点①） | serving 层的 `_should_check_for_unstreamed_tool_arg_tokens` | 提供"理论值"作为比较基准 |
| `streamed_args_for_tool` | parser 的 XML 状态机（站点③④） | 同上 | 提供"实际已发"值，与理论值比对 |

**本质**: 同一条信息（tool call 的 arguments）的两个视图：
- `prev_tool_call_arr` = 模型输出经 parser 解析后的**完整预期**
- `streamed_args_for_tool` = 已通过 SSE delta 送达客户端的**实际进度**

当 finish_reason 到达时，两者的差 = 需要补发的 delta。
