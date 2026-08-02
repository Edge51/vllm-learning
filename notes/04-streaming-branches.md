# Streaming Branch Architecture

`chat_completion_stream_generator` — 流式输出的分支决策树。

## 入口判断变量

```
is_mistral_grammar_path = request._grammar_from_tool_parser
```

由 Mistral 的 `grammar_from_tool_parser` 触发。Mistral 不走 JSON 解析，走自定义 `grammar` 约束解码。

---

```
tool_choice_function_name = request.tool_choice.function.name  # 或 None
```

`tool_choice: {type: "function", function: {name: "get_weather"}}` 时取到名称，否则 None。

---

```
tool_choice_auto = (not tool_choice_function_name
                    and request.tools
                    and self.tool_parser
                    and self.enable_auto_tools
                    and request.tool_choice in ["auto", None])
```

"auto" 模式 + 有 tool_parser + enable_auto_tools = True。启用后 `previous_text` 持续追踪，后续交给 parser 解析。

---

```
tool_choice_uses_parser = (self.tool_parser is not None
                           and not self.tool_parser.supports_required_and_named
                           and request.tools
                           and (tool_choice == "required"
                                or isinstance(tool_choice, NamedToolChoice)))
```

**关键解读**：tool_choice 是 required/named，但 parser 不能输出 JSON（如 GLM XML 格式），所以降级到 parser 路径。这个 flag 让 required/named 跳过 JSON 解析分支，落入 `parser.parse_delta()`。

---

## 3 个分类

| 分类 | tool_choice | parser 注册 | 走哪条路径 |
|------|-------------|-------------|-----------|
| **auto** | "auto"/None | 有 | `parser.parse_delta()` |
| **named** | `{function: {name: "..."}}` | — | `extract_tool_call_required_streaming()` + named-specific 逻辑 |
| **required** | "required" | — | `extract_tool_call_required_streaming()` |
| **required (XML)** | "required" | 有且 `supports_required_and_named=False` | `parser.parse_delta()` (因 `tool_choice_uses_parser` 降级) |

## 分支决策树

主循环每步迭代、每个 `output.index` 独立执行以下分支（mutually exclusive）：

```
            ┌─ use_harmony (GPT-OSS ── extract_harmony_streaming_delta()
            │  模型, token 级状态机)
            │
            ├─ is_mistral_grammar_path ── MistralToolParser
            │                            .extract_maybe_reasoning_and_tool_streaming()
            │                            (grammar 约束解码, 一次调用同时输出 reasoning + tool)
            │
            ├─ tool_choice_function_name ── named tool_choice 路径
            │   AND not tool_choice_uses_parser
            │   ├─ reasoning_parser active → extract_reasoning_streaming()
            │   └─ else → 构造 DeltaToolCall {id, type, name, arguments=delta_text}
            │             (首次生成 id+name, 后续只有 arguments)
            │
            ├─ tool_choice == "required" ── required tool_choice 路径
            │   AND not tool_choice_uses_parser
            │   ├─ reasoning_parser active → extract_reasoning_streaming()
            │   └─ else → extract_tool_call_required_streaming()
            │             (partial_json_loads 增量解析 JSON tool array)
            │
            ├─ parser is not None ── parser.parse_delta()
            │   (统一入口: DelegatingParser 内部按 reasoning_phase →
            │    tool_call_phase 顺序处理)
            │
            └─ else ── DeltaMessage(content=delta_text)
                       (无任何 parser, 纯文本流)
```

### 条件覆盖矩阵

哪个 `tool_choice` 值落入哪个分支：

| tool_choice | 无 tool_parser | 有 tool_parser | 有 parser (非 tool_parser) |
|------------|----------------|----------------|---------------------------|
| None | else | auto path | else |
| "auto" | else | auto path | else |
| "required" | required path | required 或 (if supports_required_and_named=False) parser path | required path |
| named func | named path | named 或 (if supports_required_and_named=False) parser path | named path |

## 核心模式差异

### 路径 A: `extract_tool_call_required_streaming()` (JSON 解析)

```python
# lines 430-528
def extract_tool_call_required_streaming(self, previous_text, current_text, ...):
    obj, _ = partial_json_loads(current_text, flags=Allow.ALL)
    # 对 JSON array 增量解析
    # 每次迭代只输出 diff (delta text)
    # 用 _filter_delta_text() 提取增量
```

**实现**：用 `partial_json_loads` 对整个 `current_text` 做部分 JSON 解析，然后用 `_filter_delta_text()` 从 `current_text` 里减去 `previous_text` 得到增量。

**适合**：JSON output 格式的 parser（Hermes、原生 tool call）。

### 路径 B: `parser.parse_delta()` (DelegatingParser)

```python
# abstract_parser.py:585-639
def parse_delta(self, delta_text, delta_token_ids, request, prompt_token_ids):
    state = self._stream_state
    current_text = state.previous_text + delta_text
    current_token_ids = state.previous_token_ids + delta_token_ids

    # Phase 1: reasoning
    if self._in_reasoning_phase(state):
        delta_message = self.extract_reasoning_streaming(...)
        # 如果 reasoning 结束, 转交剩余文本给 tool parser

    # Phase 2: tool call
    if self._in_tool_call_phase(state):
        delta_message = self.extract_tool_calls_streaming(...)

    # 更新 state
    state.previous_text = current_text
    state.previous_token_ids = current_token_ids
    return delta_message
```

**实现**：内部用 `StreamState` 维护 `previous_text`，每次收到 `delta_text` 拼接到 `previous_text` 上，两次调用之间的 state 是自持的。

**适合**：XML/output 格式的 parser、需要推理和工具调用共存的场景。

### 路径 C: named function path (直接构造)

```python
# lines 858-941
# 不调用任何 parser 方法
delta_tool_call = DeltaToolCall(
    id=tool_call_id,
    type="function",
    function=DeltaFunctionCall(name=tool_choice_function_name, arguments=delta_text),
)
delta_message = DeltaMessage(tool_calls=[delta_tool_call])
```

**实现**：直接取 `delta_text` 作为 `arguments` 值，首次生成 `id` 和 `name`，之后只发 `arguments`。

**适合**：`tool_choice: {function: {name: "get_weather"}}`，模型只生成参数，函数名已知。

## 各分支依赖的前置条件

| 分支 | 需要 `previous_texts` | 需要 `all_previous_token_ids` |
|------|----------------------|-------------------------------|
| harmony | 否 (token 级) | 否 |
| mistral | 是 | 是 |
| named+not_uses_parser | 是 | 是 |
| required+not_uses_parser | 是 | 是 |
| auto/uses_parser | 是 | 是 |
| parser.parse_delta() | 是 (parser 内部 self._stream_state) | 否 (由 parser 自行维护) |
| else (纯文本) | 否 | 否 |

`previous_texts` 和 `all_previous_token_ids` 仅在 `is_mistral_grammar_path`、`tool_choice_auto`、`tool_choice_uses_parser`、`reasoning_parser` 之一为 True 时初始化 (line 598-609)。

## 更新 previous_texts 的时机 (line 1017-1031)

```python
if (is_mistral_grammar_path or tool_choice_auto or tool_choice_uses_parser or reasoning_parser) and not use_harmony:
    previous_texts[i] = current_text       # 完整替换 (因为已经拼接了 delta)
    all_previous_token_ids[i] = current_token_ids
else:
    previous_texts[i] += delta_text        # 简单追加
```

auto/uses_parser/mistral/reasoning 路径用 `current_text = previous_text + delta_text` 再赋值回 `previous_texts[i]`；其他路径直接 `+= delta_text`。效果等价，但 auto 路径需要同时维护 `current_token_ids`。

## `_filter_delta_text` 的作用

```python
# lines 400-428
def _filter_delta_text(delta_text, previous_text):
    # 当 spec decode 产出多个 token 时,
    # delta_text 可能跨越了完整的 token 边界,
    # 导致重复发送已发过的 text
    obj, zero = partial_json_loads(delta_text, flags=Allow.ALL)
    return updated_delta, passed_zero
```

仅在 `extract_tool_call_required_streaming()` 内部使用，用于处理 speculative decoding 场景——draft tokens 批处理后，`delta_text` 可能包含多个 token 的文本，需要做完整 JSON 解析后再计算 diff。

## 时序图

```
serving.py                      parser (DelegatingParser)    tool_parser (XML)  /  LLM
   │                                      │                        │              │
   ├─ request.tool_choice?                 │                        │              │
   │  ├─ "required" + supports_json ──→ extract_tool_call_required_streaming()    │
   │  │                                      │                        │
   │  ├─ "required" + XML parser ──────→ parse_delta() ──→ extract_tool_calls_streaming()
   │  │                                      │                        │
   │  ├─ named + supports_json ────────→ named branch (直接构造)      │
   │  │                                      │                        │
   │  ├─ auto ─────────────────────────→ parse_delta() ──→ extract_tool_calls_streaming()
   │  │                                      │                        │
   │  └─ no tools ─────────────────────→ DeltaMessage(content=d)      │
   │                                      │                        │
   └─ 发送 SSE chunk                     │                        │
```

## 变量命名批评

| 变量名 | 问题 | 更好的名字 |
|--------|------|-----------|
| `tool_choice_uses_parser` | "uses parser" 不传达 "在 required/named 时降级到 parser 路径" 的含义 | `required_named_fallsback_to_parser` 或 `tool_choice_parser_path` |
| `tool_choice_auto` | 字面意思对，但和 `tool_choice == "auto"` 很像，实际包含了 `enable_auto_tools` 条件 | `auto_tool_parsing_enabled` |
| `is_mistral_grammar_path` | 模型名写在变量名里 | `grammar_from_tool_parser` (已有的字段名) |
| `tool_choice_function_name` | 变量名暗示它是 bool，实则是 str\|None | `named_tool_choice_func_name` |

## 相关文件

- `vllm/entrypoints/openai/chat_completion/serving.py`: 主分支逻辑 (530-1060)
- `vllm/parser/abstract_parser.py`: Parser / DelegatingParser 基类 (58-678)
- `vllm/parser/parser_manager.py`: ParserManager 注册机制 (23-)
