# Parameter Golf — Strategy & Ideas

## Hard Constraints

| Constraint | Value | Notes |
|---|---|---|
| **Max artifact size** | 16,000,000 bytes (decimal) | `train_gpt.py` code + int8/int5/int6+zlib/zstd compressed weights |
| **Max training time** | 10 minutes wallclock | 8×H100 SXM (80GB each) |
| **Eval dataset** | FineWeb validation, first 50k docs | Fixed, cannot be changed |
| **Metric** | val_bpb (bits-per-byte) | Lower is better. Tokenizer-agnostic. |
| **Tokenizer** | 1024-token SentencePiece BPE (default) | Can change, but bpb adjusts via byte LUT |
| **No external downloads at eval** | Everything must be in the artifact | No API calls, no HF downloads |
| **Code limit** | train_gpt.py ≤ 1500 lines | Hard stop in the rules |
| **Deadline** | April 30, 2026 | $1M in compute credits at stake |

## Budget Math

At different precisions, 16MB buys you:

| Precision | Bytes/param | Max params | Effective after overhead |
|---|---|---|---|
| FP32 | 4.0 | 4.0M | ~3.5M (scales, code, metadata) |
| FP16 | 2.0 | 8.0M | ~7.5M |
| INT8 | 1.0 | 16.0M | ~14M |
| INT6 | 0.75 | 21.3M | ~19M |
| INT5 | 0.625 | 25.6M | ~22M |
| INT4 | 0.5 | 32.0M | ~28M |
| Ternary (1.58-bit) | 0.2 | 80.0M | ~70M |

The SOTA uses mixed int5 (MLP) + int6 (attention) + FP16 (embeddings) ≈ ~21M effective params.

Zstd-22 compression gives an additional ~1.5x ratio on int5/int6 data vs zlib-9.

## Current Leaderboard (top entries in our fork)

| Rank | Entry | val_bpb | Layers | Dim | MLP | Key Tricks |
|---|---|---|---|---|---|---|
| 1 | 10L Int5-MLP BigramHash SWA | **1.14276** | 10 | 512 | 3x | Int5 MLP, int6 attn, BigramHash(10240), SWA(0.4), OrthoInit, SmearGate, U-Net skip, zstd-22 |
| 2 | Int6 MLP3x SmearGate BigramHash SWA | **1.14582** | 9 | 512 | 3x | Int6 uniform, SmearGate, BigramHash(4096), SWA(0.5), OrthoInit, Muon WD=0.04, zstd-22 |
| 3 | SmearGate OrthoInit MuonWD Int6 | **1.15560** | 9 | 512 | 3x | SmearGate, OrthoInit, Int6 STE QAT, U-Net skip, sliding window |
| 4 | 11L Int5-MLP MuonWD0.4 Zstd22 SWA | **1.15015** | 11 | 512 | 3x | 11 layers funded by int6, Muon WD=0.04 |
| 5 | 10L Int6 QAT Zstd MLP2.6x | **1.15862** | 10 | 512 | 2.6x | STE int6 QAT (zero quant gap), zstd-22 |
| — | Baseline (naive) | **1.22437** | 9 | 512 | 2x | Int8+zlib, no tricks |

**Gap to beat: 1.14276 → need < 1.142**

## What Everyone Is Doing (Converged Bag of Tricks)

Every top-5 submission uses some combination of:

1. **Int5/Int6 mixed quantization** — MLP in int5, attention in int6, embeddings in FP16
2. **Zstd-22** — ~5% better compression than zlib-9 on low-bit data
3. **3x MLP expansion** — hidden=1536, single largest architectural contributor
4. **10-11 layers** — funded by quantization byte savings
5. **SmearGate** — learned per-dim sigmoid gate blending current + previous token (~512 params)
6. **BigramHash** — hash consecutive token pairs into embedding table (4096-10240 buckets, dim=128)
7. **SWA** — stochastic weight averaging over last 40-50% of training, every 50 steps
8. **OrthoInit** — orthogonal initialization, pairs well with Muon
9. **Muon WD=0.04, momentum=0.99** — decoupled weight decay on all param groups
10. **Sliding window eval stride=64** — ~0.034 bpb free gain (pure eval strategy)
11. **Grad clipping 0.3** — prevents outliers that hurt quantization
12. **Seq2048 training** — 2x baseline context, ~0.012-0.023 bpb gain
13. **U-Net skip connections** — encoder half stores activations, decoder half reuses in reverse

**Key insight: nobody has changed the fundamental architecture.** They're all playing the same game — squeeze more of the same GPT into 16MB via better compression.

## What Nobody Has Tried

### Tier 1: High-Confidence Novel Ideas

#### 1. Depth Recurrence (Weight Sharing)
Train K unique blocks, loop them N times = K×N effective layers but only K layers of parameters.

**Why it works:**
- ALBERT (Google, 2019) proved this at scale
- ModernALBERT + MoL (Dec 2025) fixed the expressivity gap via per-loop LoRA
- Relaxed Recursive Transformers (2024) — shared backbone + layer-wise LoRA
- MoEUT (NeurIPS 2024) — Universal Transformers + MoE
- Mixture-of-Recursions (DeepMind 2025) — dynamic per-token recursion depth

**Budget impact:**
- 6 unique blocks × 2 loops = 12 effective layers, paying for 6
- Saves ~4.5MB at int5/int6 → fund dim=640 or MLP 4x or both
- Per-loop differentiation: learned scaling per block per loop (~12 params) or tiny LoRA (rank=4, ~8K params total)

**Risk:** Shared blocks may underperform unique blocks by 0.01-0.02 bpb. The question is whether wider dim / more MLP compensates.

**Implementation:** ~30 lines of code change in GPT.forward()

#### 2. TrigramHash (extend BigramHash to 3-grams)
Current BigramHash looks at token pairs (t_{i-1}, t_i). TrigramHash looks at (t_{i-2}, t_{i-1}, t_i).

**Why it works:**
- Web text is highly repetitive at the 3-gram level
- Captures common patterns like "the → quick → brown", "http → :// → www"
- BigramHash already gives ~0.01 bpb; trigrams should add another ~0.005

**Budget impact:**
- 16384 buckets × 128 dim = 2M params × 0.625 bytes (int5) = ~1.25MB
- Or share the same hash table as bigram (zero extra params, just different hash function)

**Risk:** Diminishing returns vs bigrams. Hash collisions increase with 3-gram space.

#### 3. Learnable Weight Clipping (per-layer quantization bounds)
Current: fixed `INT8_CLIP_PERCENTILE = 99.99984` for all layers.
Proposed: learn optimal clipping bounds per layer during QAT.

**Why it works:**
- OmniQuant (ICLR 2024) showed learnable clipping recovers 0.5-1% quality
- Different layers have wildly different weight distributions
- Early layers and attention layers are more sensitive than MLP layers
- The current fixed percentile wastes precision on layers that don't need it

**Budget impact:** Zero — just changes the quantization grid, not the model size.

**Risk:** Minimal. Strictly better than fixed clipping.

#### 4. Compression-Aware Weight Regularization
Add a differentiable penalty that encourages weights to cluster around fewer distinct quantized values, improving zstd compression ratio.

**Concrete approach:**
```python
# During training, add to loss:
def compression_reg(model, lambda_reg=0.001):
    penalty = 0
    for p in model.parameters():
        if p.ndim == 2 and p.numel() > 65536:
            # Encourage weights to cluster (low entropy after quantization)
            q = torch.round(p / (p.abs().max() / 127)) # fake quantize
            # Penalize spread of unique values
            penalty += q.float().var()
    return lambda_reg * penalty
```

**Why it works:**
- Zstd achieves better ratios when the byte stream has lower entropy
- Weight distributions that cluster around fewer values compress ~10-20% better
- Combined with WD=0.04, this pushes weights toward a compact, compressible distribution

**Budget impact:** Could save 0.5-1.5MB → room for 1 more layer or wider MLP.

**Risk:** Over-regularization could hurt model quality. Need to tune lambda.

### Tier 2: Medium-Confidence Ideas

#### 5. Sparse Outlier Preservation
Store 99% of weights in int5 but keep the top 1% highest-magnitude weights separately in FP16 as a sparse matrix.

**Why it works:**
- Outlier weights disproportionately affect quality when quantized
- SqueezeLLM and SpQR papers show 2-3% quality recovery from preserving outliers
- FP16 sparse matrix is tiny: 1% of 20M params = 200K × 2 bytes = 400KB

**Budget impact:** 400KB cost for potentially significant quality gain.

#### 6. Weight Sorting Before Compression
Before zstd compression, sort/permute weight matrix rows by magnitude or by similarity (via greedy nearest-neighbor ordering). Adjacent similar values compress much better.

**Why it works:**
- Zstd uses LZ77 + Huffman — benefits from repeated byte patterns
- Sorted weights have smoother deltas between adjacent values
- Delta encoding on sorted weights: store first value + deltas (often 0 or ±1)
- Reported 5-15% size reduction in compression benchmarks

**Budget impact:** 5-15% size savings = 0.8-2.4MB freed. Zero compute cost.

**Risk:** Must store the permutation order (or use a deterministic sort that can be reconstructed). Row-level sort is free; within-row sort needs a small index.

#### 7. Cross-Layer KV Sharing
Share K and V projection weights across all layers. Each layer keeps its own Q projection.

**Why it works:**
- CommonKV (2025) showed adjacent layers have similar KV representations
- In GQA with 4 KV heads, KV projections are already small (dim×kv_dim = 512×256 = 128K params each)
- Sharing across 10 layers saves ~2.5M params (10 layers × 2 projections × 128K)

**Budget impact:** Saves ~1.5MB at int6. Could fund an extra layer or wider dim.

**Risk:** Quality may degrade more than the param savings justify. Need to ablate.

#### 8. Custom Tokenizer on FineWeb
Train a new 1024-token SentencePiece on FineWeb training data specifically.

**Why it works:**
- Current tokenizer may not optimally cover FineWeb's byte distribution
- Better byte coverage → fewer tokens per byte → lower bpb mechanically
- The eval metric uses a byte LUT that adjusts for tokenizer differences, so it's fair

**Risk:** The rules say "submissions that edit the tokenizer will be examined more carefully." Could be disqualified if perceived as gaming the metric.

#### 9. Int4 QAT for MLP Weights
Push MLP quantization from int5 to int4 [-8, 7] with STE-based QAT.

**Why it works:**
- MLP weights are the most compressible (relu² activation creates sparse patterns)
- Int4 saves ~20% over int5 per MLP weight
- With 3x MLP (1536 hidden), MLP params dominate: 2 × 512 × 1536 = 1.57M per layer
- 10 layers × 1.57M × (0.625 - 0.5) bytes = ~1.96MB saved

**Budget impact:** ~2MB saved → fund 12th layer or dim=576.

**Risk:** Int4 may cause quality collapse even with QAT. Needs careful STE schedule and possibly per-channel scales instead of per-row.

### Tier 3: Speculative / High-Risk Ideas

#### 10. Mixture-of-Experts (MoE) MLP
Replace the 3x MLP with 2 smaller expert MLPs (each 2x) with top-1 routing.

**Budget impact:** Similar total params to 3x MLP, but tokens see different computation paths → more capacity.

**Risk:** Router adds complexity, and 10-min training may not be enough for the router to converge. MoE benefits usually appear at larger scales.

#### 11. Ternary (1.58-bit) Weights via BitNet
Train with {-1, 0, 1} weights. 16MB → ~80M params.

**Risk:** Extreme quality loss at this scale. BitNet results are promising for >1B params but unproven at 20M. Training dynamics are completely different.

#### 12. Test-Time Training (LoRA TTT)
Already attempted (1.1929 bpb, rank 11). Could be combined with all SOTA tricks for a better result.

**Risk:** TTT adds eval time. Must fit within constraints. Current TTT entry didn't combine with BigramHash/SmearGate/SWA.

#### 13. Progressive Layer Growing
Start with 6 layers for the first 60% of training, then add 4 more layers (initialized from interpolation of existing layers) for the final 40%.

**Why:** Earlier training is "wasted" on the final architecture — growing lets you iterate faster in early training, then refine with the full model.

**Risk:** Initialization of new layers is tricky. May cause training instability.

#### 14. Knowledge Distillation (Self-Distillation)
Can't download a teacher, BUT you can train a larger model in the first 5 minutes, then distill into the submission model for the last 5 minutes.

**Why:** The 8×H100 cluster has ~640GB VRAM. You can train a ~500M param teacher in FP16 for 5 minutes, then distill logits into the 20M param student.

**Risk:** 5 minutes may not be enough for either the teacher or the distillation. Very tight on time budget.

#### 15. Procedural/Fractal Weight Patterns
Generate some weight matrices from a small PRNG seed + formula. Stores almost nothing.

**Risk:** Completely untested. Likely terrible quality.

## The "Recycler" Plan (Our Submission Strategy)

### Core Thesis
**Depth recurrence + wider model + all SOTA tricks = more effective compute per byte than anyone else.**

Everyone else: 10 unique blocks × dim=512 → ~21M params → 16MB
Us: 6 unique blocks × 2 loops × dim=640 → 12 effective layers, ~15M unique params → 16MB but wider and deeper

### Architecture

```
Model: "The Recycler"
  - 6 unique transformer blocks, looped 2× = 12 effective layers
  - dim = 640 (25% wider than SOTA's 512)
  - MLP 3× expansion (hidden = 1920), relu²
  - 10 heads, 5 KV heads (GQA), head_dim = 64
  - SmearGate + BigramHash(10240, dim=128)
  - U-Net skip connections (adapted for 12 effective layers)
  - Per-loop learned residual scaling (2 × dim params per block per loop = 7680 total)
  - Tied embeddings, vocab=1024
  - logit_softcap = 20.0 (tighter for int5/int6)
  - RoPE base = 10000 (or 50000 for seq2048)
```

### Parameter Budget Estimate

| Component | Params | Bytes (mixed int5/int6) |
|---|---|---|
| Embedding (1024×640) | 655K | ~82KB (FP16, tiny) |
| 6 blocks × attention (Q+K+V+O, 640→640) | 6 × 1.64M = 9.8M | ~7.4MB (int6) |
| 6 blocks × MLP (640→1920→640) | 6 × 2.46M = 14.7M | ~9.2MB (int5) |
| BigramHash table (10240×128) | 1.3M | ~0.8MB (int6) |
| BigramHash proj (128→640) | 82K | ~0.05MB |
| Skip weights, scales, norms | ~50K | ~0.1MB (FP16) |
| Per-loop scaling (12 × 640 × 2) | 15K | ~0.03MB (FP16) |
| **Total** | **~26.6M** | **~17.6MB raw** |

After zstd-22 compression (typical 1.3-1.5x ratio on int5/int6): **~12-13.5MB**

That's **well under 16MB** with room to spare. Could push to dim=704 or MLP 3.5×.

### Training Recipe

```
Optimizer: Muon (matrix_lr=0.02, WD=0.04, momentum=0.99, warmup 0.92→0.99 over 1500 steps)
           AdamW for scalars/embeddings (WD=0.04)
Batch: 786K tokens/step, seq_len=2048
Iterations: 20000 (wallclock capped at 600s)
Warmup: 20 steps
Warmdown: 3000 iters (cosine)
Grad clip: 0.3
QAT: STE int5 (MLP) + int6 (attention), start at 75% of training, ramp over 5%
SWA: start_frac=0.4, every=50 steps
Init: Orthogonal + muP-scaled output projections
3% magnitude pruning before final quantization
```

### Eval

```
Sliding window: stride=64, seq_len=2048
Compression: zstd-22
```

### Incremental Validation Plan

| Step | What | Expected bpb | Risk |
|---|---|---|---|
| 0 | Reproduce SOTA baseline (10L int5) | 1.1428 | None — sanity check |
| 1 | Depth recurrence at dim=512 (6 blocks × 2) | ~1.150? | Recurrence penalty |
| 2 | Widen to dim=640 | ~1.140? | May compensate |
| 3 | Add per-loop scaling | ~1.138? | Should help expressivity |
| 4 | Add TrigramHash | ~1.135? | Incremental |
| 5 | Learnable clipping bounds | ~1.133? | Free improvement |
| 6 | Compression-aware regularization | ~1.130? | If size permits |
| 7 | Weight sorting before zstd | ~size savings | Frees room for more params |

### Fallback Plan
If depth recurrence underperforms (step 1-2 worse than expected):
- Abandon recurrence, go with 11L dim=512 MLP 3x (like submission #4)
- Stack TrigramHash + learnable clipping + compression reg on top of SOTA
- Target incremental win: 1.140 → 1.135

## Quick Reference: Key Constants to Tune

| Constant | Current SOTA | Range to Sweep | Impact |
|---|---|---|---|
| `num_layers` | 10 | 10-12 (with recurrence: 6×2, 4×3) | High |
| `model_dim` | 512 | 512-704 (with recurrence savings) | High |
| `mlp_mult` | 3.0 | 2.6-4.0 | High |
| `num_heads` / `num_kv_heads` | 8/4 | 10/5 (at dim=640) | Medium |
| `bigram_vocab_size` | 10240 | 10240-32768 | Medium |
| `swa_start_frac` | 0.4 | 0.3-0.5 | Medium |
| `warmdown_iters` | 3000 | 2500-4000 | Medium |
| `weight_decay` | 0.04 | 0.03-0.06 | Medium |
| `muon_momentum` | 0.99 | 0.98-0.995 | Low |
| `grad_clip_norm` | 0.3 | 0.2-0.5 | Low |
| `logit_softcap` | 30.0 | 15-30 (lower for int5) | Low |
| `rope_base` | 10000 | 10000-100000 (for seq2048) | Low |
| `train_batch_tokens` | 786432 | 524288-1048576 | Low |
| `INT8_CLIP_PERCENTILE` | 99.99984 | Per-layer learned | Medium |

## Infra Options

| Platform | GPUs | Cost | Notes |
|---|---|---|---|
| RunPod | 8×H100 SXM | ~$25/hr | Already set up in repo (Flash framework) |
| PrimeIntellect | 8×H100 | ~$20/hr | Used for previous H100 sweep |
| Modal | 8×H100 | ~$30/hr | Used for PEFT paper |
| GCP Vertex AI | 8×A100 80GB | Quota approved | Slower than H100, but free credits |

Each run = 10 minutes = ~$4-5 per experiment. Budget for 20 experiments = ~$80-100.

## Files to Modify

- `train_gpt.py` — the single submission file (architecture + training + quantization + eval)
- Keep under 1500 lines
- Everything must be self-contained (no imports beyond standard + torch + numpy + sentencepiece)
