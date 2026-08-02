# Block Lifecycle: 1024 seq_len request 的生死线

vLLM v0.20.2 · block_size = 16

---

## 调用链图

```
Scheduler.schedule()
  │
  ├─ Running (line 467)
  │   └─ kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
  │       [kv_cache_manager.py:265]
  │       │
  │       ├─ coordinator.get_num_blocks_to_allocate()        ← 先算要多少
  │       │   [kv_cache_coordinator.py:80]
  │       │   └─ SingleTypeKVCacheManager.get_num_blocks_to_allocate()
  │       │       [single_type_kv_cache_manager.py:88]
  │       │
  │       ├─ 检查 pool 够不够: num_blocks > block_pool.get_num_free_blocks()?
  │       │   [kv_cache_manager.py:395]
  │       │   └─ 不够 → return None → scheduler 开始踢人
  │       │
  │       └─ coordinator.allocate_new_blocks()
  │           [kv_cache_coordinator.py:163]
  │           └─ SingleTypeKVCacheManager.allocate_new_blocks()
  │               [single_type_kv_cache_manager.py:242]
  │               └─ block_pool.get_new_blocks(num_new_blocks)
  │                   [block_pool.py:322]
  │                   └─ FreeKVCacheBlockQueue.popleft_n(num_blocks)
  │                       [kv_cache_utils.py:248]
  │
  ├─ Waiting (line 758)
  │   └─ 同上 allocate_slots（首次 prefill 走这里）
  │
  └─ Preempt (line 506 → 974)
      └─ kv_cache_manager.free(request)
          [kv_cache_manager.py:437]
          └─ coordinator.free(request_id)
              [kv_cache_coordinator.py:211]
              └─ SingleTypeKVCacheManager.free()
                  [single_type_kv_cache_manager.py:303]
                  └─ block_pool.free_blocks(ordered_blocks)
                      [block_pool.py:408]
                      └─ FreeKVCacheBlockQueue.append_n()
                          [kv_cache_utils.py:327]
```

---

## Part 1 — Block 怎么生

### 1. 真正的分配入口：BlockPool.get_new_blocks()

**文件**: `block_pool.py:322-352`

```python
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    if num_blocks > self.get_num_free_blocks():
        raise ValueError(...)

    ret = self.free_block_queue.popleft_n(num_blocks)

    for block in ret:
        assert block.ref_cnt == 0
        block.ref_cnt += 1    # ★ 分配时 ref_cnt 置为 1
    return ret
```

- **参数**: `num_blocks: int` — 要几个 block
- **返回**: `list[KVCacheBlock]` — 从 free list 头部 pop 出来的 block
- **ref_cnt**: 拿出来的 block `ref_cnt = 0 → 1`，标记为"被占用"

### 2. 调用链（自顶向下）

```
scheduler.py:467 / 758  (Running / Waiting)
  → kv_cache_manager.py:265  KVCacheManager.allocate_slots()
    → kv_cache_coordinator.py:163  allocate_new_blocks()
      → single_type_kv_cache_manager.py:242  allocate_new_blocks()
        → block_pool.py:322  get_new_blocks()
          → kv_cache_utils.py:248  FreeKVCacheBlockQueue.popleft_n()
```

### 3. 一次分配多少 block？

`single_type_kv_cache_manager.py:259-261`:

```python
req_blocks = self.req_to_blocks[request_id]
num_required_blocks = cdiv(num_tokens, self.block_size)
num_new_blocks = num_required_blocks - len(req_blocks)
```

- `cdiv` = ceiling division，定义在 `vllm/utils/math_utils.py:10`
- 新 request: `len(req_blocks) = 0`，`num_new_blocks = cdiv(num_tokens, block_size)`
- 已有 block 的 request（decode 阶段）: 只在 `cdiv` 进位时才触发（每 16 tokens 一次）

**1024 seq_len / 16 block_size 新请求**: `num_new_blocks = cdiv(1024, 16) = **64**`

---

## Part 2 — Block 怎么管

### 4. BlockTable 的数据结构

**文件**: `single_type_kv_cache_manager.py:70-73`

```python
self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)
```

- **不是 HashMap / dict 结构**，是 **ordered list（Vec）**
- `req_to_blocks[request_id]` = `list[KVCacheBlock]` — 按逻辑 block 索引顺序排列
- 元素类型 `KVCacheBlock`（`kv_cache_utils.py:114`），关键字段:
  - `block_id: int` — 物理 block ID（0 ~ num_gpu_blocks-1）
  - `ref_cnt: int` — 引用计数
  - `_block_hash` — 用于 prefix caching 的 hash

### 5. Logical → Physical 映射

**没有显式的映射表。** 映射就是 **list index → block.block_id**:

```python
req_blocks = self.req_to_blocks["req_A"]
logical_block_3 = req_blocks[3]
physical_id = req_blocks[3].block_id  # ← 物理 block ID
```

逻辑连续性由 list 的下标连续性保证，物理 block_id 可以不连续。

### 6. BlockTable 存在哪

`SingleTypeKVCacheManager.req_to_blocks[request_id]`（每个 `SingleTypeKVCacheManager` 实例持有自己的 `req_to_blocks`）。

标准 LLM（Llama / Qwen）只有一个 `FullAttentionManager`，所以一个 request 只有一个 block table。混合模型（attention + mamba）才有多个。

---

## Part 3 — Block 怎么死

### 7. Preempt 时谁调了 free？

**文件**: `scheduler.py:965-985`

```python
def _preempt_request(self, request, timestamp):
    self.kv_cache_manager.free(request)     # (1) 释放所有 block
    request.num_computed_tokens = 0          # (2) 重置进度
    request.status = RequestStatus.PREEMPTED
    self.waiting.prepend_request(request)    # (3) 放回 waiting 队列
```

调用链:

```
scheduler.py:506  self._preempt_request(preempted_req, ...)
  → scheduler.py:974  self.kv_cache_manager.free(request)
    → kv_cache_manager.py:437  free(request)
      → kv_cache_coordinator.py:211  free(request_id)
        → single_type_kv_cache_manager.py:303  free(request_id)
          → block_pool.py:408  free_blocks(ordered_blocks)
```

### 8. free 之后 block 回 free list 还是直接还给 GPU？

**回 free list（FIFO 双向链表）**，不是还给 GPU。

`single_type_kv_cache_manager.py:303-318`:

```python
def free(self, request_id):
    req_blocks = self.req_to_blocks.pop(request_id, [])   # 从字典移除
    ordered_blocks = reversed(req_blocks)                  # 倒序释放
    self.block_pool.free_blocks(ordered_blocks)
```

`block_pool.py:408-422`:

```python
def free_blocks(self, ordered_blocks):
    blocks_list = list(ordered_blocks)
    for block in blocks_list:
        block.ref_cnt -= 1                                # ref_cnt -1
    self.free_block_queue.append_n(
        [block for block in blocks_list
         if block.ref_cnt == 0 and not block.is_null]     # 仅 ref_cnt=0 才回 pool
    )
```

- `free_block_queue` 是 `FreeKVCacheBlockQueue`（`kv_cache_utils.py:162`），一个**双向链表**
- `append_n()`（`kv_cache_utils.py:327`）把 block 插到链表尾部
- 分配时 `popleft()` 从头部取 → **FIFO 顺序**

**ref_cnt 的关键作用**: 如果 prefix caching 使一个 block 被多个 request 共享（ref_cnt > 1），只释放一个 request 时 ref_cnt 减为 0 但不等于 0，该 block **不会回 free list**，直到最后一个引用的 request 释放它。

### 9. 空闲 block 数量在哪查？

`block_pool.py:478-484`:

```python
def get_num_free_blocks(self) -> int:
    return self.free_block_queue.num_free_blocks
```

使用示例（分配前的检查，`kv_cache_manager.py:395`）:

```python
if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
    return None   # 不够，分配失败 → scheduler 踢人
```

---

## 1024 token 计算示例

### 场景

| 参数 | 值 |
|---|---|
| sequence length | 1024 |
| block_size | 16 |
| max_model_len | ≥ 1024 |
| 无 prefix cache 命中 |

### 分配阶段（Waiting → Running）

| 步骤 | 代码 | 结果 |
|---|---|---|
| scheduler 算 `num_new_tokens` | `scheduler.py:677` | `1024 - 0 = 1024` |
| `get_num_blocks_to_allocate()` | `single_type_kv_cache_manager.py:119` | `cdiv(1024, 16) = 64` |
| 检查 free list | `kv_cache_manager.py:395` | `64 <= pool_size ✅` |
| `get_new_blocks(64)` | `block_pool.py:336` | 从 free list pop 64 blocks |
| ref_cnt 置为 1 | `block_pool.py:343` | 每个 block: `0 → 1` |
| `req_blocks.extend(new_blocks)` | `single_type_kv_cache_manager.py:266` | block table = [b0, b1, ..., b63] |

**分配结果**: 64 blocks, 64 个物理 page, 1024 token 的 KV cache 全部可写。

### Decode 阶段（运行中）

每 decode 1 个 token，`num_new_tokens = 1`。前 16 个 decode token 不需要新 block（`cdiv(1025, 16) = 64`, `64 - 64 = 0`）。第 17 个 decode token 触发 `cdiv(1040, 16) = 65`，需要再分配 1 个 block。以此类推。

### Preempt 阶段

| 步骤 | 代码 | 结果 |
|---|---|---|
| `_preempt_request()` | `scheduler.py:974` | `kv_cache_manager.free(request)` |
| `req_to_blocks.pop()` | `single_type_kv_cache_manager.py:311` | 取出全部 64 blocks |
| `reversed()` | line 315 | 倒序: [b63, b62, ..., b0] |
| `ref_cnt -= 1` | `block_pool.py:419` | 每个 block: `1 → 0` |
| `ref_cnt == 0` 过滤 | line 421 | 全部 64 个通过 |
| `append_n()` | line 420 | 64 个 block 回到 free list 尾部 |

**释放结果**: 64 个 block 全部回 free list，可被后续请求复用。

### 如果有 prefix cache 命中（3 个共享 block）

假设 64 个 block 中 block[0], block[1], block[2] 被另一个 request 共享（ref_cnt = 2）。

释放时: `ref_cnt -= 1` → `2 → 1`。`ref_cnt == 0` 条件不满足，这 3 个 block **不回 free list**。剩下 61 个正常回去。那 3 个 block 要等另一个 request 也释放时才真正归还。

---

## 检验答案

> 一个 1024 seq_len 的请求，满 block 16 一次 prefill，分配了多少 block？

**64 个 block**。`cdiv(1024, 16) = 64`。

> 怎么分配的？

通过 `SingleTypeKVCacheManager.allocate_new_blocks()`，先 `cdiv` 算出需要 64，当前 `req_to_blocks` 为空（新请求），`num_new_blocks = 64`。调 `block_pool.get_new_blocks(64)`，从 `FreeKVCacheBlockQueue` 双向链表头部 pop 64 个，每个 `ref_cnt = 0 → 1`。然后 `req_blocks.extend()` 挂到 block table 尾部。

> Preempt 时 block 去了哪里？

`_preempt_request()` → `kv_cache_manager.free(request)` → `req_to_blocks.pop(request_id)` 取下全部 64 个 → `reversed` 倒序 → `block_pool.free_blocks()` 逐个 `ref_cnt -= 1` → `ref_cnt == 0` 的通过 `append_n()` 插回 `FreeKVCacheBlockQueue` 链表尾部。如果有其他 request 共享了某些 block（ref_cnt > 1），它们不回去，等所有引用都释放才回。
