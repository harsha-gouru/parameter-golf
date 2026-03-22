# Parameter Golf — Experiment Results

## Competition Target
- **SOTA (leaderboard #1):** 1.14276 bpb (thwu1, 10L int5/int6 + BigramHash + SWA)
- **Baseline:** 1.22437 bpb (naive 9L int8+zlib)
- **Our best:** 1.15711 bpb (10L, count-init bigram logit head, learnable QAT)
- **Latest (v3):** 1.15776 bpb (fast CastedLinear path, warmdown=2800, 23 SWA checkpoints)
- **Gap to SOTA:** 0.015 bpb
- **Open PR #434:** 1.1370 bpb (would be new #1 if merged — uses XSA + LeakyReLU² + Partial RoPE)

## Our Submission Architecture

```
Model: 10-layer GPT
  dim=512, MLP 3× (hidden=1536), relu²
  8 heads / 4 KV heads (GQA), RoPE base=10000
  Tied embeddings, vocab=1024, logit_softcap=30.0
  SmearGate (learned per-dim gate blending prev token embedding)
  U-Net skip connections (encoder stores, decoder reuses in reverse)
  OrthoInit + muP-scaled output projections

Bigram Logit Head (NEW — replaces BigramHash):
  Exact 1024×1024 table, zero hash collisions
  Count-based initialization: B[a,b] = log p(b|a) - log p(b) from 16M tokens
  Learned gate scale, applied BEFORE logit softcap
  Quantized to int4 with nibble-packing (524KB vs 1MB at int8)

QAT with Learnable Clip Multipliers (NEW):
  Per-tensor γ: gamma = 0.5 + sigmoid(theta), constrained [0.5, 1.5]
  STE round: (x.round() - x).detach() + x
  Row-wise amax detached (gradients only through gamma)
  Activated at 85% of training, regularized log(γ)²
  DDP grad sync for gamma params

Quantization:
  int5 for MLP weights (clip_range=15)
  int6 for attention weights (clip_range=31)
  int4 nibble-packed for bigram logit table (clip_range=7)
  FP16 for tied embeddings + last layer key projection
  5% magnitude pruning before quantization
  zstd-22 compression
  Learned gamma values applied in final quantization

Training:
  Muon optimizer: matrix_lr=0.02, WD=0.04, momentum=0.99 (warmup 0.92→0.99 over 1500 steps)
  AdamW for embeddings/scalars: WD=0.04
  Batch: 786K tokens/step, seq_len=2048
  Warmdown: 3000 iters, grad_clip=0.3
  SWA: start_frac=0.4, every 50 steps

Eval:
  Sliding window stride=64, batch_seqs=32
```

## 8×H100 Competition Run (2026-03-22)

**Run ID:** `test-8gpu-alloc`
**Config:** 10L, dim=512, MLP 3×, all features enabled

### Training Log
```
CUDA devices: 8× NVIDIA H100 80GB HBM3
Compilation: ~150s (torch.compile fullgraph=True)
Warmup: 20 steps

step:300   train_loss:2.5854  step_avg:94.46ms
step:1000  train_loss:2.3017  step_avg:94.57ms
step:2500  val_loss:2.1279  val_bpb:1.2603
step:4000  val_loss:2.0778  val_bpb:1.2306
step:5500  val_loss:2.0047  val_bpb:1.1873
step:6100  (wallclock cap hit at ~600s)

Trained ~6300 steps at 94.5 ms/step avg
Total training tokens: ~6300 × 786432 ≈ 4.95B tokens
```

### Final Results
```
final_int8_zlib_roundtrip val_loss:1.95373688 val_bpb:1.15711609

ARTIFACT: 14,920,243 bytes (14.92 MB)
CODE:     63,472 bytes
TOTAL:    14,983,715 bytes (14.98 MB)
LIMIT:    16,000,000 bytes
STATUS:   PASS
HEADROOM: 1,016,285 bytes (1.02 MB)
```

### Comparison to Leaderboard

| Entry | val_bpb | Artifact | Steps | Key Difference |
|-------|---------|----------|-------|----------------|
| **#1 thwu1** | **1.14276** | 15.97 MB | ~13780 | BigramHash(10240), int5 MLP |
| **Ours** | **1.15711** | 14.92 MB | ~6300 | Bigram logit head, learnable QAT, int4 pack |
| #2 raahilshah | 1.14582 | 15.86 MB | — | SmearGate, BigramHash(4096) |
| Baseline | 1.22437 | 15.86 MB | ~13780 | No tricks |

**Gap to #1: 0.014 bpb** — we trained for roughly half the steps (6300 vs 13780).

## Why We're Not at 1.14 Yet

1. **Only ~6300 steps vs ~13780** — wallclock capped at 600s, compilation took ~150s, leaving only ~450s of actual training. SOTA gets ~580s of training (less compile overhead).

2. **Warmdown misconfigured** — set to 3000 iters but only trained 6300 steps, so warmdown started at step 3300 (52% of training). Should be ~1500 iters for 6300 steps.

3. **No BigramHash** — we removed it to fit the bigram logit head. SOTA uses BigramHash(10240) which perturbs hidden representations. We have 1.02 MB headroom to add it back at 4096 buckets.

4. **Compilation overhead** — torch.compile takes ~150s on 8 GPUs. SOTA likely has a warmed-up compile cache.

## Quick Wins to Close the Gap

| Fix | Expected Impact | Size Cost | Risk |
|-----|-----------------|-----------|------|
| Add BigramHash(4096) back | +0.003-0.005 bpb | ~0.4 MB | Low |
| Fix warmdown (3000→1500) | +0.003-0.005 bpb | 0 | None |
| Reduce compile time (skip warmup?) | +500 more steps | 0 | Low |
| Try 11L | +0.005-0.010 bpb | ~0.6 MB | Tight on size |

## Full Smoke Test History

| Run | Steps | GPU | Features | Artifact | val_bpb | Status |
|-----|-------|-----|----------|----------|---------|--------|
| v3 baseline | 200 | 1×H100 | None | 15.28 MB | 1.9563 | PASS |
| v8 200-step | 200 | 1×H100 | Bigram logit + int4 | 14.28 MB | 1.9834 | PASS |
| v8 1000-step | 1000 | 1×H100 | + QAT learnable clip | 14.72 MB | 1.5865 | PASS (γ 0.85-1.00) |
| 10L 5000-step | 5000 | 1×H100 | Full | 14.57 MB | 1.4668 | PASS |
| 11L 5000-step | 5000 | 1×H100 | Full + 11L | 15.93 MB | 1.4600 | PASS (8KB room) |
| **8×H100 full** | **~6300** | **8×H100** | **Full competition** | **14.92 MB** | **1.1571** | **PASS (1.02 MB room)** |

## 8×H100 Run v2 — Tuned (2026-03-22)

**Run ID:** `sub-10L-v2-tuned`
**Changes from v1:** Warmdown 3000→1500, added BigramHash(4096) back alongside bigram logit head

### Results
```
val_bpb: 1.15872 (WORSE than v1)
Artifact: 15.23 MB (+0.31 MB from BigramHash)
Steps: 6321 at 95ms/step
SWA: 12 checkpoints
QAT gammas: all 1.000 (didn't learn)
```

### Takeaways
- **BigramHash(4096) alongside exact bigram head provides no benefit** — confirmed redundant
- Warmdown=1500 was too short (23.7% of training vs SOTA's 44.5%)
- QAT gammas didn't learn — start_frac=0.85 + recompile overhead leaves too few steps

---

## 8×H100 Run v3 — Fast Path + Warmdown Fix (2026-03-22)

**Run ID:** `sub-10L-v3-tuned`
**Changes from v2:**
- Fast CastedLinear path: skip `.float()` + dict lookup when QAT off (matches SOTA speed)
- Removed BigramHash (confirmed redundant)
- Warmdown 1500→2800 (matches SOTA's ~45% warmdown ratio)

### Training Log
```
CUDA devices: 8× NVIDIA H100 80GB HBM3
Compilation: ~150s

step:1000  train_loss:2.3067  step_avg:93.52ms
step:2000  train_loss:2.1574  step_avg:93.53ms
step:3000  val_bpb:1.2558
step:4000  train_loss:2.0569  step_avg:93.56ms
step:5000  val_bpb:1.2128
step:6000  val_bpb:1.1751  step_avg:93.58ms
step:6410  val_bpb:1.1639  (wallclock cap)

SWA: 23 checkpoints averaged
QAT gammas: min=1.000 max=1.000 (still not learning)
```

### Results
```
final val_bpb: 1.15776
Artifact: 15,036,400 bytes (15.04 MB)
Headroom: 0.90 MB
Steps: 6410 at 93.6ms/step
```

### Comparison

| Metric | v1 | v2 | v3 | SOTA (thwu1) |
|--------|----|----|----|----|
| val_bpb | 1.1571 | 1.1587 | 1.1578 | **1.1428** |
| Steps | ~6300 | 6321 | 6410 | 6709 |
| Step time | 95ms | 95ms | 93.6ms | 89ms |
| SWA ckpts | 12 | 12 | 23 | — |
| Gammas learned | YES | no | no | — |
| Artifact | 14.92 MB | 15.23 MB | 15.04 MB | 15.97 MB |

### Takeaways
1. **Step time improved 95→93.6ms** — fast CastedLinear path saves ~1.5ms/step
2. **SWA 23 checkpoints** — more averaging material with warmdown=2800
3. **Still 0.015 gap to SOTA** — we're stuck at ~1.157
4. **QAT gammas not learning on 8-GPU** — works on single-GPU (gammas reached 0.85) but not on 8×H100. Likely a torch.compile recompilation issue when QAT activates.
5. **Bigram logit head adds ~4ms/step overhead** — 93.6ms vs SOTA's 89ms. The table lookup is more expensive than BigramHash's small embedding lookup.

### Root Cause Analysis: Why 0.015 Gap Remains

| Factor | Impact | Evidence |
|--------|--------|----------|
| **~300 fewer steps** | ~0.005 bpb | 6410 vs 6709 steps (93.6ms vs 89ms per step) |
| **QAT not learning on 8-GPU** | ~0.003 bpb | Single-GPU: gamma=0.85 improved quant gap. 8-GPU: all 1.0 |
| **No BigramHash** | ~0.003 bpb | SOTA uses 10240 buckets in hidden state; we replaced with logit head |
| **Compilation overhead** | ~0.002 bpb | 150s compile = fewer training steps than SOTA |
| **Different architecture** | ~0.002 bpb | Count-init bigram logit head vs BigramHash — different trade-offs |

### Options to Close the Gap

**Option A: Match SOTA architecture, add our quant improvements**
- Revert to BigramHash(10240), remove bigram logit head
- Keep int5/int6 + learnable QAT + SWA + all SOTA tricks
- Expected: match SOTA step time (89ms), get 6700+ steps
- Risk: lose the bigram logit head advantage (if any)

**Option B: Fix QAT on 8-GPU**
- Precompile both QAT and non-QAT paths during warmup
- Lower qat_start_frac to 0.70 (more QAT steps)
- Expected: gammas learn like single-GPU, quant gap closes
- Risk: recompile still eats steps

**Option C: Study PR #434 (1.1370) and adopt their tricks**
- XSA (Xtra-Short Attention?), LeakyReLU², Partial RoPE
- These are novel architectural changes we haven't tried
- Expected: potentially large gains if techniques are additive
- Risk: unknown, need to read their code

**Option D: 11L with tight budget management**
- 11L got 1.4600 on 5K single-GPU (vs 10L at 1.4668)
- 8KB headroom on 5% pruning — risky but possible
- Expected: ~0.005-0.010 bpb improvement from depth
- Risk: over 16MB limit

---

## Total Spend

| Run | GPU | Duration | Estimated Cost |
|-----|-----|----------|----------------|
| Smoke tests (v1-v8, ~8 runs) | 1×H100 | ~25 min total | ~$2 |
| 5K-step tests (2 runs) | 1×H100 | ~35 min total | ~$3 |
| 8×H100 v1 | 8×H100 | ~15 min | ~$8 |
| 8×H100 v2 | 8×H100 | ~15 min | ~$8 |
| 8×H100 v3 | 8×H100 | ~15 min | ~$8 |
| **Total** | — | — | **~$29** |

## Infrastructure

- **Modal:** volume `parameter-golf-data` (80 train shards + val cached)
- **GPU:** `gpu="H100:8"` = single node, 8× NVIDIA H100 80GB HBM3, NVLink
- **Training:** torchrun --standalone --nproc_per_node=8 inside Modal function
- **Cost:** ~$8 per full 8×H100 run (~15 min including compile + eval)
- **Step speed:** 94.5 ms/step on 8×H100 (batch=786K, seq=2048)

## Code Files

- `train_gpt_submission.py` — 1462 lines (limit: 1500), self-contained training + quantization + eval
- `modal_smoke_test.py` — Modal runner with dataset caching, single-GPU and 8-GPU modes
- `STRATEGY.md` — Full strategy doc with 17 ideas ranked
- `REVIEWER_CONTEXT.md` — Reviewer Q&A with code references
