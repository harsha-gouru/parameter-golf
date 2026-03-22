# Parameter Golf — Full Context for Review

## What This Is
OpenAI competition: train the best LM in ≤16MB artifact, ≤10 min on 8×H100 SXM. Scored on val_bpb (bits-per-byte, FineWeb validation). Lower is better.

## Current Leaderboard (merged)
| Rank | Score | Author |
|------|-------|--------|
| #1 | 1.1428 | thwu1 |
| #2 | 1.1458 | Raahil Shah |
| #3 | 1.1502 | aruniyer |

## Pending PRs (not yet merged, would be new top)
| PR | Score | Key Techniques |
|----|-------|----------------|
| #401 | **1.1243** | 11L, XSA4, Partial RoPE 16/64, LN Scale, EMA(0.997), BigramHash(2048), LR=0.025 |
| #315 | **1.1248** | Same stack. Confirmed QAT is dead-code-eliminated by torch.compile |
| #434 | **1.1370** | 10L, XSA4, Partial RoPE, LeakyReLU², LR=0.025, BigramHash(10240) |

## Our Best Result
**val_bpb = 1.1551** (8×H100, 10 min, run `sub-10L-v4-xsa`)

## Our Architecture
```
10-layer GPT, dim=512, MLP 3× (hidden=1536), relu²
8 heads / 4 KV heads (GQA), tied embeddings, vocab=1024
SmearGate, U-Net skip connections, OrthoInit + muP scaling
logit_softcap=30.0

NEW vs SOTA baseline:
- XSA (Exclusive Self Attention) on last 4 layers
- Partial RoPE: 16 of 64 head dims get rotary, rest position-free
- LN Scale: block output *= 1/sqrt(layer_idx + 1)
- Higher LR: matrix=0.025, scalar=0.025, tied_embed=0.035
- Exact BigramLogitHead with count-based initialization (replaces BigramHash)
- Int4 nibble-packed bigram table (pack_i4/unpack_i4)
- 8% magnitude pruning (up from 3%)

Quantization: int5 MLP, int6 attention, int4 bigram table, FP16 embeddings, zstd-22
Training: Muon (WD=0.04, momentum=0.99), warmdown=2800, SWA(0.4, every 50), seq=2048, batch=786K
Eval: sliding window stride=64
```

## 8×H100 Run History (v1→v4)

### v1: Bigram logit head + count-init (first real run)
```
val_bpb: 1.1571 | artifact: 14.92 MB | headroom: 1.02 MB
steps: 6300 | step_avg: 95ms
Features: count-init BigramLogitHead, int4 packed, no BigramHash, QAT with learnable clip
Result: QAT gammas learned on single-GPU (0.85) but NOT on 8-GPU (all 1.000)
```

### v2: + BigramHash(4096) + warmdown=1500
```
val_bpb: 1.1587 (WORSE) | artifact: 15.23 MB | headroom: 0.71 MB
steps: 6321 | step_avg: 95ms
Result: BigramHash alongside BigramLogitHead provided NO benefit. Confirmed redundant.
```

### v3: Fast CastedLinear path + warmdown=2800
```
val_bpb: 1.1578 | artifact: 15.04 MB | headroom: 0.90 MB
steps: 6410 | step_avg: 93.6ms
Changes: Skip .float() + dict lookup in CastedLinear when QAT off. Warmdown=2800 (matches SOTA 45% ratio).
Result: Step time improved 95→93.6ms (+110 more steps). SWA collected 23 checkpoints. QAT still dead.
```

### v4: + XSA + Partial RoPE + LN Scale + higher LR (CURRENT BEST)
```
val_bpb: 1.1551 | artifact: 14.81 MB | headroom: 1.13 MB
steps: 6225 | step_avg: 96.4ms
Changes: Adopted top-PR architectural tricks. Removed all QAT code (confirmed dead by PR #315).
Result: 0.003 bpb gain from architecture despite fewer steps (XSA adds compute).
Convergence: step 5000→1.2019, step 5500→1.1839, step 6000→1.1661, step 6225→1.1603 (cap)
Post-SWA+quant: 1.1551
```

## Gap Analysis: Us (1.1551) vs Top Pending (1.1248)

| Factor | Impact | What they have that we don't |
|--------|--------|------------------------------|
| **11 layers** | ~0.010 bpb | PR #315/#401 use 11L. We have 1.13 MB headroom — enough |
| **EMA (0.997)** | ~0.005 bpb | Exponential moving average of weights, stacked with SWA |
| **Fewer steps** | ~0.003 bpb | They get 7051 steps at 85ms. We get 6225 at 96ms |
| **BigramHash(2048)** | ~0.002 bpb | We use exact bigram logit head instead (different trade-off) |
| **LeakyReLU²** | ~0.003 bpb | PR #434 uses it. We still use relu² |

## What We Tried That Didn't Work

1. **QAT with learnable clip multipliers** — torch.compile constant-folds the branch, gammas never learn on 8-GPU. Confirmed by PR #315. Removed.
2. **BigramHash + BigramLogitHead together** — redundant, no improvement, +0.31 MB waste.
3. **TrigramHash** — hash formula was mathematically broken (token cancellation). Removed.
4. **Naive weight row-sorting** — breaks model function without compensating permutations. Removed.
5. **Compression regularization** — torch.round has zero gradient; smooth version is just L-infinity reg, marginal at best. Removed.

## What We Haven't Tried Yet

1. **11 layers** (1.13 MB headroom available)
2. **EMA** (~30 lines, stacks with SWA)
3. **LeakyReLU(0.5)²** instead of relu²
4. **Sensitivity-based bit allocation** (calibrate int4/5/6 per tensor)
5. **Post-quant temperature calibration**
6. **FlashAttention 3** (Hopper-specific, PR #315 uses it)

## Key Code Sections

### Our unique features (not in any other PR):

**BigramLogitHead with count-init** — exact 1024×1024 table initialized from corpus bigram log-probabilities, applied BEFORE softcap:
```python
class BigramLogitHead(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(vocab_size, vocab_size, dtype=torch.float32))
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, prev_tokens: Tensor) -> Tensor:
        return self.table[prev_tokens] * self.scale

# Count-init before training:
def build_bigram_residual_init(pattern, vocab_size, sample_tokens=16_000_000, alpha=0.25, clip_value=4.0):
    stream = TokenStream(pattern)
    counts = torch.zeros(vocab_size * vocab_size, dtype=torch.int64)
    # ... count bigrams from first 16M tokens ...
    C = counts.view(vocab_size, vocab_size).float()
    log_p_cond = torch.log(C + alpha) - torch.log(row_sum + alpha * vocab_size)
    log_p_uni = torch.log(col_sum + alpha) - torch.log(total + alpha * vocab_size)
    B = (log_p_cond - log_p_uni).clamp_(-clip_value, clip_value)
    return B

# Applied in GPT.forward BEFORE softcap:
if self.bigram_logit is not None:
    logits_proj = logits_proj + self.bigram_logit(input_ids.reshape(-1))
logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
```

**Int4 nibble packing** for bigram table (524KB vs 1MB at int8):
```python
def pack_i4(q: Tensor) -> Tensor:
    qu = (q.view(-1).to(torch.int16) & 0xF).to(torch.uint8)
    if qu.numel() & 1:
        qu = torch.cat([qu, torch.zeros(1, dtype=torch.uint8, device=qu.device)])
    return (qu[0::2] | (qu[1::2] << 4)).contiguous()

def unpack_i4(packed: Tensor, numel: int) -> Tensor:
    p = packed.view(-1).to(torch.uint8)
    lo = (p & 0x0F).to(torch.int16)
    hi = ((p >> 4) & 0x0F).to(torch.int16)
    q = torch.empty(p.numel() * 2, dtype=torch.int16, device=p.device)
    q[0::2] = lo; q[1::2] = hi; q = q[:numel]
    q = torch.where(q >= 8, q - 16, q)
    return q.to(torch.int8)
```

### Architectural changes (adopted from top PRs):

**XSA on last 4 layers** — removes self-value component from attention output:
```python
# In CausalSelfAttention.forward, after scaled_dot_product_attention:
if self.use_xsa:
    vn = F.normalize(v, dim=-1)
    if self.num_kv_heads != self.num_heads:
        vn = vn.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
    y = y - (y * vn).sum(dim=-1, keepdim=True) * vn
```

**Partial RoPE (16/64 dims)** — rotary on 25% of head dims:
```python
# Rotary module sized to rope_dims=16 (not full head_dim=64)
# In forward:
rd = self.rope_dims  # 16
if rd < self.head_dim:  # 16 < 64
    q_rot = apply_rotary_emb(q[..., :rd], cos, sin)
    k_rot = apply_rotary_emb(k[..., :rd], cos, sin)
    q = torch.cat([q_rot, q[..., rd:]], dim=-1)
    k = torch.cat([k_rot, k[..., rd:]], dim=-1)
```

**LN Scale** — damp deeper layers:
```python
# In GPT.forward, after each block:
if self.ln_scale:
    x = x * (1.0 / math.sqrt(i + 1))
```

## Questions for You

1. **Should we go 11L?** We have 1.13 MB headroom. PR #315 gets 1.1248 with 11L. Our 10L is at 1.1551. The extra layer could close ~0.010 bpb of the 0.030 gap.

2. **Should we add EMA?** PR #315/#401 both use EMA(0.997) stacked with SWA. ~30 lines of code. But we're at 1394/1500 lines — tight.

3. **Is our BigramLogitHead actually helping vs BigramHash?** The top PRs all use BigramHash (2048-10240 buckets), none use an exact bigram head. Our head adds ~4ms/step overhead (96ms vs ~89ms for SOTA without it = ~200 fewer steps). Is the count-init advantage worth 200 lost steps?

4. **Step time: 96ms vs 85ms (PR #315).** PR #315 gets 7051 steps at 85ms. We get 6225 at 96ms. That's 826 more steps for them. Sources of our overhead:
   - BigramLogitHead table lookup: ~4ms
   - XSA compute: ~3ms
   - What else? Is it the LN Scale multiply? Count-init pre-pass?

5. **LeakyReLU(0.5)² vs relu²** — PR #434 claims ~0.003 bpb from this alone. Should we switch? It's a 1-line change.

6. **Any other techniques we're missing?** PR #401 mentions "Shared Value Embedding (dim=128, layers 9,10)" and "FlashAttention 3 (Hopper)". Are these worth pursuing?

7. **What would YOU change in our code to close the remaining 0.030 gap to 1.1248?**

## Full Code
The full `train_gpt_submission.py` (1394 lines) is in the repo. Key file locations:
- Hyperparameters: lines 40-103
- Int4 pack/unpack: lines 505-533
- CastedLinear: lines 535-539
- Rotary + apply_rotary_emb: lines 549-576
- CausalSelfAttention (XSA + Partial RoPE): lines 579-637
- BigramLogitHead + count-init: lines 689-733
- Block (with XSA/RoPE params): lines 736-753
- GPT (with LN Scale): lines 758-877
- mixed_quantize_int6 (with int4 path): lines 356-410
- Training loop: lines 1215-1310
- Magnitude pruning + quantization export: lines 1340-1395
