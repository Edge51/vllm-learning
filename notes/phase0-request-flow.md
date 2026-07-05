# Phase 0: vLLM 请求处理流程

```mermaid
flowchart LR
    A["HTTP 请求<br>（包含 tools）"] --> B["jinja 渲染<br>（tools → prompt 模板）"]
    B --> C["tokenize<br>（文本 → token IDs）"]
    C --> D["加入调度队列"]
    D --> E["scheduler 调度<br>（资源检查 + 选请求）"]
    E --> F{"新请求？"}
    F -->|是| G["prefill<br>批量计算 KV cache"]
    F -->|否| H["decode<br>逐 token 生成"]
    G --> I["post_process<br>（收集输出 token）"]
    H --> I
    I --> J{"生成完成？<br>（eos / max_tokens）"}
    J -->|否| E
    J -->|是| K["detokenize<br>（token IDs → 文本）"]
    K --> L["API 层组装响应"]
    L --> M["返回客户端"]

    style A fill:#e1f5fe
    style M fill:#c8e6c9
    style E fill:#fff3e0
    style G fill:#f3e5f5
    style H fill:#f3e5f5
```

## 各环节对应代码

| 环节 | 文件 |
|---|---|
| API 入口 | `vllm/entrypoints/openai/api_server.py` |
| 引擎主循环 | `vllm/v1/engine/core.py` |
| 调度器 | `vllm/v1/core/sched/scheduler.py` |
| 模型执行 | `vllm/v1/worker/gpu/model_runner.py` |
