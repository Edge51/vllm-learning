# third_party — Git 子模块索引

此目录包含上游官方项目的 git 子模块，配置为浅克隆 (`--depth 1`)。

| 子模块 | 仓库地址 | 版本 | 当前锁定提交 |
|---|---|---|---|
| `third_party/vllm` | https://github.com/vllm-project/vllm.git | tag `v0.29.0` | `98dff2a8` |
| `third_party/vllm-ascend` | https://github.com/vllm-project/vllm-ascend.git | tag `v0.23.0` | `5cb98caa` |
| `third_party/vllm-cloud` | https://gitcode.com/InferTurbo/vllm-cloud.git | branch `qwen_glm` | `2d720c65` |
| `third_party/ops-transformer` | https://gitcode.com/cann/ops-transformer.git | tag `v9.2.0` | `dff22133` |

> 注意：vllm / vllm-ascend / ops-transformer 锁定稳定 release 标签，避免 master 分支版本不匹配风险。

使用方式：

```bash
# 首次拉取子模块
git submodule update --init --depth 1

# 更新所有子模块到最新
git submodule update --remote --depth 1
```
