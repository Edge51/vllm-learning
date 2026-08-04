# Transformer Layer Matrix Computation - ASCII Diagrams

## 1. TP Column Split (ColumnParallelLinear)

```
                  W_gate (8192, 28672)  Full Matrix
+--------------------------------------------------+
|##################################################|
|##################################################|  8192
|##################################################|
+--------------------------------------------------+
                          |
                  Split along columns (TP=4)
                          v
+----------+----------+----------+----------+
|  rank 0  |  rank 1  |  rank 2  |  rank 3  |
| (8192,   | (8192,   | (8192,   | (8192,   |
|  7168)   |  7168)   |  7168)   |  7168)   |  8192
|          |          |          |          |
+----------+----------+----------+----------+
   7168       7168       7168       7168
                          |
                  Each card computes independently
                          v
+----------+----------+----------+----------+
| partial  | partial  | partial  | partial  |
| output 0 | output 1 | output 2 | output 3 |
| (1,2048, | (1,2048, | (1,2048, | (1,2048, |
|  7168)   |  7168)   |  7168)   |  7168)   |
+----------+----------+----------+----------+
                          |
                     all-gather
                  (concat results)
                          v
                  +-------------------+
                  |  Full Output      |
                  |  (1, 2048, 8192)  |
                  +-------------------+
```

**Key insight**: Input x is full (1, 2048, 8192). Each card has full input. Weight is split, each card computes part of columns. all-gather concatenates 4 x 7168 back to 8192.

---

## 2. TP Row Split (RowParallelLinear)

```
 W_down (28672, 8192)                 x also split along rows
+----------+                      Each card: (1, 2048, 2048)
|##########|
|##########| rank 0              +--------+
|##########| (7168, 8192)        |  x0    |--> partial sum 0
+----------+                     +--------+
|##########|
|##########| rank 1              +--------+
|##########| (7168, 8192)        |  x1    |--> partial sum 1
+----------+                     +--------+
|##########|
|##########| rank 2              +--------+
|##########| (7168, 8192)        |  x2    |--> partial sum 2
+----------+                     +--------+
|##########|
|##########| rank 3              +--------+
|##########| (7168, 8192)        |  x3    |--> partial sum 3
+----------+                     +--------+
    |                                 |
    +---------------------------------+
                      |
                 all-reduce
              (sum: 0+1+2+3)
                      |
                      v
            +-------------------+
            |  Full Output      |
            |  (1, 2048, 8192)  |
            +-------------------+
```

**Key insight**: Weight split along rows (input dim 28672 -> each card 7168). Input x also split along rows. Each card computes partial sum. all-reduce adds 4 partial sums = full result.

---

## 3. Column vs Row Comparison

```
+-------------------------------------------+  +-------------------------------------------+
| ColumnParallelLinear                      |  | RowParallelLinear                         |
+-------------------------------------------+  +-------------------------------------------+
|                                           |  |                                           |
|  x (full input)                           |  |  x (split along rows)                     |
|     |                                     |  |     |                                     |
|     v                                     |  |     v                                     |
|  [W_chunk]  (split along columns)         |  |  [W_chunk]  (split along rows)            |
|     |                                     |  |     |                                     |
|     v                                     |  |     v                                     |
|  partial output  (each card has 1/N)      |  |  partial sum  (each card computes sum)    |
|     |                                     |  |     |                                     |
|     v                                     |  |     v                                     |
|  all-gather  (concat N parts)             |  |  all-reduce  (sum N parts)                |
|     |                                     |  |     |                                     |
|     v                                     |  |     v                                     |
|  full output  (1, seq, hidden)            |  |  full output  (1, seq, hidden)            |
|                                           |  |                                           |
+-------------------------------------------+  +-------------------------------------------+
```

**In Attention**:
- W_q, W_k, W_v -> ColumnParallel (split cols, output partial)
- W_o -> RowParallel (split rows, output partial sum -> all-reduce)

**In FFN**:
- W_gate, W_up -> ColumnParallel (split cols)
- W_down -> RowParallel (split rows -> all-reduce)

---

## 4. Attention Matrix Multiplication

```
    Q (64, 2048, 128)           K^T (64, 128, 2048)
+-----------------+          +-----------------+
|  heads=64       |          |  heads=64       |
|  seq=2048       |          |  head_dim=128   |
|  head_dim=128   |          |  seq=2048       |
+-----------------+          +-----------------+
        |                            |
        +------------x---------------+
                       |
                       v
              +-------------------+
              |  scores           |
              |  (64, 2048, 2048) |
              |  per-head attn    |
              |  matrix           |
              |  scores[i][j] =   |
              |  attn weight      |
              |  token_i -> j     |
              +-------------------+
                       |
                  / sqrt(128) + causal mask
                       |
                       v
                     softmax
                       |
                       v
    V (64, 2048, 128)          scores (64, 2048, 2048)
+-----------------+          +-----------------+
|  heads=64       |          |  scores         |
|  seq=2048       |          |  (2048, 2048)   |
|  head_dim=128   |          |                 |
+-----------------+          +-----------------+
        |                            |
        +------------x---------------+
                       |
                       v
              +-------------------+
              |  output           |
              |  (64, 2048, 128)  |
              |                   |
              |  reshape ->       |
              |  (1, 2048, 8192)  |
              +-------------------+
```

**GQA (Grouped Query Attention)**:
- Q has 64 heads, K/V only 8 heads
- Every 8 Q heads share 1 KV head
- K/V repeat_interleave(8) to expand to 64 heads

---

## 5. FFN (SwiGLU)

```
    hidden_states (1, 2048, 8192)
                 |
         +-------+-------+
         v               v
    +---------+    +---------+
    | W_gate  |    | W_up    |
    | (8192,  |    | (8192,  |
    |  28672) |    |  28672) |
    +---------+    +---------+
         v               v
    +---------+    +---------+
    |  gate   |    |   up    |
    | (1,2048,|    | (1,2048,|
    |  28672) |    |  28672) |
    +---------+    +---------+
         v               v
         +-------+-------+
                 v
          SiLU(gate) * up
                 v
          +-------------+
          |  activated  |
          | (1, 2048,   |
          |  28672)     |
          +-------------+
                 |
         +-------+-------+
         v
    +---------+
    | W_down  |
    | (28672, |
    |  8192)  |
    +---------+
         v
    +-------------+
    |  output     |
    | (1, 2048,   |
    |  8192)      |
    +-------------+
```

**FFN pattern**: up-project (8192 -> 28672), then down-project (28672 -> 8192). Middle dim is 3.5x.

---

## 6. Full Transformer Layer

```
    token_ids (1, 2048)
         |
         v
    +-------------+
    |  Embedding  |  nn.Embedding(vocab, 8192)
    +-------------+
         |
         v
    hidden_states (1, 2048, 8192)
         |
         v
    +-------------------------------------------+
    |           Transformer Layer x 80           |
    |                                           |
    |  +---------+                              |
    |  | RMSNorm |                              |
    |  +---------+                              |
    |       |                                   |
    |       v                                   |
    |  +-------------------------------------+  |
    |  |        Attention (GQA)              |  |
    |  |  Q = x @ W_q  (ColumnParallel)      |  |
    |  |  K = x @ W_k  (ColumnParallel)      |  |
    |  |  V = x @ W_v  (ColumnParallel)      |  |
    |  |  scores = Q @ K^T                   |  |
    |  |  out = softmax(scores) @ V          |  |
    |  |  out = out @ W_o (RowParallel)      |  | <-- all-reduce
    |  +-------------------------------------+  |
    |       |                                   |
    |       v                                   |
    |  +---------+                              |
    |  |Residual |  <-- skip connection         |
    |  +---------+                              |
    |       |                                   |
    |       v                                   |
    |  +---------+                              |
    |  | RMSNorm |                              |
    |  +---------+                              |
    |       |                                   |
    |       v                                   |
    |  +-------------------------------------+  |
    |  |        FFN (SwiGLU)                  |  |
    |  |  gate = x @ W_gate (ColumnParallel)  |  |
    |  |  up = x @ W_up    (ColumnParallel)   |  |
    |  |  act = SiLU(gate) * up              |  |
    |  |  out = act @ W_down (RowParallel)   |  | <-- all-reduce
    |  +-------------------------------------+  |
    |       |                                   |
    |       v                                   |
    |  +---------+                              |
    |  |Residual |                              |
    |  +---------+                              |
    |                                           |
    +-------------------------------------------+
         |
         v
    +---------------+
    | Final RMSNorm |
    +---------------+
         |
         v
    +---------------+
    |   LM Head     |  Linear(8192, vocab)
    +---------------+
         |
         v
    logits (1, 2048, vocab)
         |
         v
    argmax -> next token
```

---

## 7. TP Communication Pattern

```
Per-layer communication:

    ColumnParallel          RowParallel
    (W_q, W_k, W_v,        (W_o, W_down)
     W_gate, W_up)
         |                       |
         v                       v
    Each card computes      Each card computes
    independently           partial sum
    (no communication)      (no communication)
         |                       |
         v                       v
    +---------+           +---------+
    | partial |           | partial |
    | result  |           | sum     |
    +---------+           +---------+
         |                       |
         v                       v
    all-gather             all-reduce
    (concat)               (sum)
         |                       |
         v                       v
    +---------+           +---------+
    | full    |           | full    |
    | output  |           | output  |
    +---------+           +---------+
```

**TP=4 means 2 cross-card communications per layer**:
1. After Attention: all-reduce (after W_o)
2. After FFN: all-reduce (after W_down)

**80 layers = 160 all-reduces per token**. This is why TP cannot cross nodes (bandwidth too low).

---

## 8. PP Split

```
    +-------------------------------------------+
    |              Stage 0 (GPU 0-3)             |
    |                                           |
    |   Layer 0 -> Layer 1 -> ... -> Layer 39   |
    |                                           |
    +-------------------------------------------+
                         |
              Pass hidden_states (1, 2048, 8192)
              Only 1 transfer per step
                         |
                         v
    +-------------------------------------------+
    |              Stage 1 (GPU 4-7)             |
    |                                           |
    |  Layer 40 -> Layer 41 -> ... -> Layer 79  |
    |                                           |
    +-------------------------------------------+
                         |
                         v
                   Final RMSNorm
                         |
                         v
                      LM Head
```

**PP vs TP communication volume**:
- TP: 2 all-reduces per layer x 80 layers = 160 synchronizations
- PP: 1 hidden_states transfer per forward pass = 1 synchronization

PP is node-friendly, TP must stay within a node.

---

## 9. Attention - Every Step with Matrix Shapes

```
Input: hidden_states (batch=1, seq=2048, hidden=8192)

============================================================
Step 1: Q, K, V Projections (ColumnParallel)
============================================================

  hidden_states (1, 2048, 8192)
         |
         +--------------------+--------------------+
         |                    |                    |
         v                    v                    v
    +----------+         +----------+         +----------+
    |   W_q    |         |   W_k    |         |   W_v    |
    | (8192,   |         | (8192,   |         | (8192,   |
    |  8192)   |         |  1024)   |         |  1024)   |
    +----------+         +----------+         +----------+
         |                    |                    |
         v                    v                    v
    Q (1, 2048, 8192)    K (1, 2048, 1024)    V (1, 2048, 1024)

  Note: W_k and W_v are smaller because GQA (8 KV heads vs 64 Q heads)

============================================================
Step 2: Reshape to (batch, heads, seq, head_dim)
============================================================

  Q (1, 2048, 8192)                 K (1, 2048, 1024)
         |                                |
         v                                v
  reshape(1, 2048, 64, 128)        reshape(1, 2048, 8, 128)
         |                                |
         v                                v
  Q (1, 64, 2048, 128)            K (1, 8, 2048, 128)
         |                                |
         v                                v
  transpose(1,2)                    transpose(1,2)
         |                                |
         v                                v
  Q (1, 64, 2048, 128)            K (1, 8, 2048, 128)

  Same for V:
  V (1, 2048, 1024) -> reshape -> V (1, 8, 2048, 128)

============================================================
Step 3: GQA - Repeat K, V to match Q heads
============================================================

  Q has 64 heads, K has 8 heads
  Every 8 Q heads share 1 KV head

  K (1, 8, 2048, 128)
         |
         v
  repeat_interleave(repeats=8, dim=1)
         |
         v
  K (1, 64, 2048, 128)   <-- now same head count as Q

  Same for V:
  V (1, 8, 2048, 128) -> repeat -> V (1, 64, 2048, 128)

============================================================
Step 4: Attention Scores = Q @ K^T
============================================================

  Q (1, 64, 2048, 128)           K^T (1, 64, 128, 2048)
  [batch, heads, seq, dim]       [batch, heads, dim, seq]
         |                              |
         +--------------x---------------+
                        |
                        v
              scores (1, 64, 2048, 2048)
              [batch, heads, seq_q, seq_k]

  Interpretation:
  scores[b][h][i][j] = how much token i attends to token j

============================================================
Step 5: Scale + Causal Mask + Softmax
============================================================

  scores (1, 64, 2048, 2048)
         |
         v
  scores = scores / sqrt(128)   <-- scale by head_dim
         |
         v
  Apply causal mask:
  +-------+-------+-------+-------+
  |  0    | -inf  | -inf  | -inf  |  token 0 can only see itself
  |  0    |   0   | -inf  | -inf  |  token 1 can see 0,1
  |  0    |   0   |   0   | -inf  |  token 2 can see 0,1,2
  |  0    |   0   |   0   |   0   |  token 3 can see all
  +-------+-------+-------+-------+
         |
         v
  softmax(dim=-1)   <-- softmax over key dimension
         |
         v
  attn_weights (1, 64, 2048, 2048)
  Each row sums to 1.0

============================================================
Step 6: Weighted Sum = attn_weights @ V
============================================================

  attn_weights (1, 64, 2048, 2048)    V (1, 64, 2048, 128)
  [batch, heads, seq_q, seq_k]        [batch, heads, seq_k, dim]
         |                                    |
         +----------------x-------------------+
                           |
                           v
              attn_output (1, 64, 2048, 128)
              [batch, heads, seq_q, dim]

  attn_output[b][h][i] = sum_j( weights[i][j] * V[j] )
  = weighted average of values, weighted by attention scores

============================================================
Step 7: Reshape back + Output Projection (RowParallel)
============================================================

  attn_output (1, 64, 2048, 128)
         |
         v
  transpose(1,2)  -> (1, 2048, 64, 128)
         |
         v
  reshape(1, 2048, 8192)   <-- concat all heads
         |
         v
  +----------+
  |   W_o    |
  | (8192,   |
  |  8192)   |
  +----------+
         |
         v
  output (1, 2048, 8192)
  This is RowParallel -> all-reduce across TP ranks
```

---

## 10. FFN - Every Step with Matrix Shapes

```
Input: hidden_states (1, 2048, 8192)   (after Attention + Residual + RMSNorm)

============================================================
Step 1: Gate and Up Projections (ColumnParallel)
============================================================

  hidden_states (1, 2048, 8192)
         |
         +--------------------+
         |                    |
         v                    v
    +----------+         +----------+
    | W_gate   |         | W_up     |
    | (8192,   |         | (8192,   |
    |  28672)  |         |  28672)  |
    +----------+         +----------+
         |                    |
         v                    v
    gate (1, 2048, 28672)  up (1, 2048, 28672)

  Both are ColumnParallel -> each card has 1/4 of columns

============================================================
Step 2: Activation = SiLU(gate) * up
============================================================

  gate (1, 2048, 28672)          up (1, 2048, 28672)
         |                              |
         v                              |
    SiLU(gate)                          |
    = gate * sigmoid(gate)              |
         |                              |
         +--------------*---------------+
                        |
                        v
              activated (1, 2048, 28672)

  Element-wise multiplication, no communication needed

============================================================
Step 3: Down Projection (RowParallel)
============================================================

  activated (1, 2048, 28672)
         |
         v
    +----------+
    | W_down   |
    | (28672,  |
    |  8192)   |
    +----------+
         |
         v
    output (1, 2048, 8192)
    This is RowParallel -> all-reduce across TP ranks
```

---

## 11. TP Block Index - How Weight Shards Map to Columns/Rows

```
Example: W_gate (8192, 28672) with TP=4

Full matrix columns:  0     7168   14336  21504  28672
                      |      |      |      |      |
                      v      v      v      v      v
+----------+----------+----------+----------+----------+
|  rank 0  |  rank 1  |  rank 2  |  rank 3  |
| cols     | cols     | cols     | cols     |
| 0-7167   | 7168-    | 14336-   | 21504-   |
|          | 14335    | 21503    | 28671    |
+----------+----------+----------+----------+
    7168       7168       7168       7168

Each rank holds: W_gate_shard (8192, 7168)

When computing: x (1, 2048, 8192) @ W_gate_shard (8192, 7168)
  -> gate_shard (1, 2048, 7168)   <-- only 1/4 of output columns

After all-gather: concat [gate_shard_0, gate_shard_1, gate_shard_2, gate_shard_3]
  -> gate (1, 2048, 28672)   <-- full output
```

```
Example: W_down (28672, 8192) with TP=4

Full matrix rows:  0     7168   14336  21504  28672
                   |      |      |      |      |
                   v      v      v      v      v
+----------+----------+----------+----------+
|  rank 0  |  rank 1  |  rank 2  |  rank 3  |
| rows     | rows     | rows     | rows     |
| 0-7167   | 7168-    | 14336-   | 21504-   |
|          | 14335    | 21503    | 28671    |
+----------+----------+----------+----------+
    7168       7168       7168       7168

Each rank holds: W_down_shard (7168, 8192)

Input x also split along rows:
  x_shard (1, 2048, 7168)   <-- only 1/4 of input rows

When computing: x_shard (1, 2048, 7168) @ W_down_shard (7168, 8192)
  -> output_shard (1, 2048, 8192)   <-- partial sum

After all-reduce: output_shard_0 + output_shard_1 + output_shard_2 + output_shard_3
  -> output (1, 2048, 8192)   <-- full result
```

---

## 12. Complete Data Flow with TP=4

```
hidden_states (1, 2048, 8192)   <-- full on all cards

============ Attention ============

  Q = x @ W_q   (ColumnParallel)
  +------------------------------------------+
  | Each card: x (1,2048,8192) @ W_q_shard  |
  |          (8192, 2048) = Q_shard          |
  |          (1, 2048, 2048)                 |
  +------------------------------------------+
           |
           v
  all-gather Q across ranks
           |
           v
  Q (1, 2048, 8192) -> reshape -> (1, 64, 2048, 128)

  K = x @ W_k   (ColumnParallel)
  V = x @ W_v   (ColumnParallel)
  Same pattern -> all-gather -> reshape

  GQA: repeat K, V to match Q heads

  scores = Q @ K^T   (independent per card, no communication)
  scores = softmax(scores / sqrt(128) + mask)
  attn_out = scores @ V   (independent per card, no communication)

  out = attn_out @ W_o   (RowParallel)
  +------------------------------------------+
  | Each card: attn_out_shard @ W_o_shard    |
  |          = partial_sum (1, 2048, 8192)   |
  +------------------------------------------+
           |
           v
  all-reduce (sum partial sums)
           |
           v
  attn_output (1, 2048, 8192)   <-- full on all cards

============ FFN ============

  gate = x @ W_gate   (ColumnParallel)
  +------------------------------------------+
  | Each card: x (1,2048,8192) @ W_gate_sh  |
  |          (8192, 7168) = gate_shard       |
  |          (1, 2048, 7168)                 |
  +------------------------------------------+
           |
           v
  all-gather gate across ranks
           |
           v
  gate (1, 2048, 28672)

  up = x @ W_up   (ColumnParallel)
  Same pattern -> all-gather -> up (1, 2048, 28672)

  activated = SiLU(gate) * up   (independent, no communication)

  out = activated @ W_down   (RowParallel)
  +------------------------------------------+
  | Each card: activated_shard @ W_down_sh   |
  |          (7168, 8192) = partial_sum      |
  |          (1, 2048, 8192)                 |
  +------------------------------------------+
           |
           v
  all-reduce (sum partial sums)
           |
           v
  ffn_output (1, 2048, 8192)   <-- full on all cards

============ Residual + RMSNorm ============

  output = RMSNorm(x + attn_output + ffn_output)
  (element-wise, no communication)
```

---

## 13. Communication Count per Layer

```
TP=4, 1 layer:

  Attention:
    1. all-gather Q   (after W_q)
    2. all-gather K   (after W_k)
    3. all-gather V   (after W_v)
    4. all-reduce out  (after W_o)
    = 4 communications

  FFN:
    1. all-gather gate  (after W_gate)
    2. all-gather up    (after W_up)
    3. all-reduce out   (after W_down)
    = 3 communications

  Total per layer: 7 communications
  Total for 80 layers: 560 communications per token

  But in practice:
  - Q/K/V projections are often fused -> 1 all-gather
  - Gate/Up projections are often fused -> 1 all-gather
  - So realistic: 4 communications per layer = 320 per token
```
