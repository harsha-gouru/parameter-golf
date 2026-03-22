# Parameter Golf — Data-Driven Analysis

## All 8×H100 Data Points

| Run | Layers | MLP | Bigram | Tricks | Steps | ms/step | Pre-avg bpb | Post-avg+quant bpb | Artifact |
|-----|--------|-----|--------|--------|-------|---------|-------------|--------------------|----|
| v1 | 10 | 3.0 | LogitHead | LR=0.02 | 6300 | 95 | 1.1647 | 1.1571 (SWA) | 14.92 MB |
| v2 | 10 | 3.0 | LogitHead+Hash4096 | LR=0.02 | 6321 | 95 | 1.1647 | 1.1587 (SWA) | 15.23 MB |
| v3 | 10 | 3.0 | LogitHead | LR=0.02, fast path | 6410 | 93.6 | 1.1639 | 1.1578 (SWA) | 15.04 MB |
| v4 | 10 | 3.0 | LogitHead | +XSA+RoPE+LNScale, LR=0.025 | 6225 | 96.4 | 1.1603 | 1.1551 (SWA) | 14.81 MB |
| v5 | **11** | **3.0** | Hash10240 | +LeakyReLU, LR=0.025 | 5700 | 105 | **1.1409** | 1.3138 (EMA broke) | 17.01 MB (OVER) |
| SOTA | 10 | 3.0 | Hash10240 | LR=0.02 | 6709 | 89 | 1.1533 | 1.1428 (SWA) | 15.97 MB |

## The Key Formula

**steps = 600s / step_time**

This is exact — the wallclock measures only main loop time, not compile/warmup.

| Config | Estimated step_time | Predicted steps |
|--------|--------------------|----|
| 10L, mlp=3.0, no bigram head | ~90ms (match SOTA) | 6667 |
| 10L, mlp=3.0, bigram logit head | ~96ms (v4 actual) | 6250 |
| 11L, mlp=3.0 | ~105ms (v5 actual) | 5714 |
| 11L, mlp=2.75 | ~100ms (estimated: 105 × 0.952) | 6000 |
| 11L, mlp=2.5 | ~95ms (estimated) | 6316 |

## Convergence Analysis

From v5 (11L, best architecture), step-by-step bpb:
```
step 5500: 1.1457
step 5700: 1.1409
rate: -0.000024 bpb/step (still dropping fast)
```

Extrapolating (with 50% decay because convergence slows):
```
step 6000: ~1.135
step 6500: ~1.127
step 6700: ~1.124
```

SWA typically improves by 0.005-0.008 (from v1-v4 data):
```
v1: 1.1647 → 1.1571 = -0.008
v3: 1.1639 → 1.1578 = -0.006
v4: 1.1603 → 1.1551 = -0.005
average SWA gain: ~0.006
```

## Projections

### Option C: 10L + BigramHash(10240) + XSA + PartialRoPE + LNScale + LeakyReLU + LR=0.025
- Step time: ~90ms → 6667 steps
- v4 was 1.1551 at 6225 steps with LR=0.025 (no LeakyReLU, had bigram logit head)
- LeakyReLU adds ~-0.003
- Replacing bigram logit with BigramHash: ~neutral (faster steps offset)
- Extra steps (6667 vs 6225): at -0.000015/step = -0.007
- **Predicted pre-SWA: ~1.145**
- **Predicted post-SWA+quant: ~1.139**

### Option D: 11L + mlp=2.75 + XSA + PartialRoPE + LNScale + LeakyReLU + LR=0.025
- Step time: ~100ms → 6000 steps
- v5 was 1.1409 at 5700 steps (11L, mlp=3.0)
- mlp=2.75 penalty: ~+0.003 (less capacity)
- Extra steps (6000 vs 5700): at -0.000024/step = -0.007
- **Predicted pre-SWA: ~1.137**
- **Predicted post-SWA+quant: ~1.131**

### Option E: 11L + mlp=2.5 + same tricks
- Step time: ~95ms → 6316 steps
- mlp=2.5 penalty: ~+0.006
- Extra steps (6316 vs 5700): at -0.000024/step = -0.015
- **Predicted pre-SWA: ~1.133**
- **Predicted post-SWA+quant: ~1.127**

## The Real Problem: We Removed SWA

We replaced SWA with EMA. EMA broke quantization on all runs. SWA was working perfectly in v1-v4 (consistent -0.006 gain).

**We must re-add SWA code before the next 8×H100 run.**

The EMA code can stay but disabled. SWA is the proven averaging method for this contest.

## Decision Matrix

| Option | Predicted bpb | Size (smoke tested) | Risk | Steps |
|--------|------------|-------|------|-------|
| **D: 11L+mlp2.75** | **~1.131** | 15.46 MB (PASS) | Medium (mlp shrink untested at scale) | ~6000 |
| C: 10L+Hash10240 | ~1.139 | 14.86 MB (PASS) | Low (proven architecture) | ~6667 |
| E: 11L+mlp2.5 | ~1.127 | Not tested | High (aggressive mlp shrink) | ~6316 |

## Critical Fix Before Next Run

**Re-add SWA alongside EMA.** EMA stays disabled, SWA enabled by default. This is the #1 blocker.

## Recommendation

**Option D (11L + mlp=2.75) with SWA re-enabled.** Predicted 1.131 would beat all merged entries and compete with pending PRs.

Smoke test already passed (15.46 MB, 0.54 MB headroom). Just need to add SWA back.
