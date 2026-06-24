# third_party — Git 子模块索引

此目录包含上游官方项目的 git 子模块，配置为浅克隆 (`--depth 1`)。

| 子模块 | 仓库地址 | 当前锁定提交 |
|---|---|---|
| `third_party/vllm` | https://github.com/vllm-project/vllm.git | `56ca599` |
| `third_party/vllm-ascend` | https://github.com/vllm-project/vllm-ascend.git | `4fbf24f` |

使用方式：

```bash
# 首次拉取子模块
git submodule update --init --depth 1

# 更新到最新
git submodule update --remote --depth 1
```
