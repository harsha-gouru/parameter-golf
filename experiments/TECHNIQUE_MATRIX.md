# Parameter Golf — Technique Matrix & Strategy
*Updated: 2026-03-19*

## Competitive Landscape (All Known Submissions)

### Merged Leaderboard (upstream/main)
| Rank | BPB    | Author       | Key Techniques |
|------|--------|-------------|----------------|
| 1    | 1.1748 | notapplica  | Sliding window, FP16 embed, 10L, Muon WD, overtone init, resid mix |
| 2    | 1.1925 | Matthew Li  | Sliding window eval stride=64 |
| 3    | 1.1928 | samacqua    | LoRA TTT (test-time training) |
| 4    | 1.2014 | Spokane Way | 4k seq length + better hypers |
| 5    | 1.2060 | Spokane Way | 2048 seq length |
| 6    | 1.2147 | Nan Liu     | 10L, mixed int8/int6 |
| 7    | 1.2197 | R. Velazco  | FP16 embed + LR/warmdown tuning |
| 8    | 1.2244 | Baseline    | 9L 512dim 1024vocab |

### Open PRs (Pending Merge — Real Competition)
| BPB    | PR#  | Author        | Key Techniques |
|--------|------|--------------|----------------|
| 1.1539 | #135 | unnir        | OrthoInit + Int6 MLP3x + BigramHash + SmearGate + zstd-22 |
| 1.1594 | #150 | yahya010     | Int6 QAT + BigramHash + MLP 1344 (WIP) |
| 1.1602 | #156 | dexhunter    | Int6 STE + NorMuon + SWA + Sliding Window |
| 1.1631 | #147 | ankitmaloo   | Smaller batch SOTA, int6+zlib |
| 1.1666 | #137 | abhishekgahlot| Int6 + MLP3x + STE QAT + NorMuon + sliding window |
| 1.1767 | #152 | timowhite88  | TTT (test-time training) |
| 1.2029 | #139 | ksang123     | BitNet b1.58 — 64.5M ternary params |
| 1.2196 | #148 | iverbovoy    | Depth Recurrence 3×4 + Cross-Repeat Skip (12.83MB) |

### Non-Record Notable
| BPB    | Track          | Key Insight |
|--------|---------------|-------------|
| 1.1574 | non-record    | Int6 + MLP3x + sliding window (merged) |
| 1.2074 | 4-hour        | Shows unlimited compute ceiling for baseline arch |

---

## Technique Taxonomy

### A. CONSTANTS (Everyone uses these — table stakes)
| Technique | Value | Why |
|-----------|-------|-----|
| Vocab size | 1024 (SP BPE) | Optimal for 16MB budget |
| Model dim | 512 | Sweet spot for param count |
| KV heads | 4 | Standard GQA |
| Tied embeddings | Yes | Saves ~0.5MB |
| Muon optimizer | Yes | Faster convergence than Adam alone |
| FineWeb 10B dataset | SP-1024 | Challenge default |
| Max wallclock | 600s | Challenge constraint |
| 8×H100 | torchrun | Challenge constraint |
| zlib/zstd compression | Yes | Required for artifact |

### B. PROVEN VARIABLES (Tunable knobs with clear winners)
| Technique | Baseline | Current Best | BPB Impact | Notes |
|-----------|----------|-------------|------------|-------|
| Num layers | 9 | 10 | -0.01 | Fits with int6 compression |
| Sliding window eval | No | stride=64 | -0.03 | Free at eval time |
| FP16 tied embed | No | Yes | -0.005 | Avoids double quant error |
| Warmdown iters | 1200 | 3000-20000 | -0.009 | Smoother weights for quant |
| Quantization | int8 | int6 per-row | -0.01 | Frees bytes for more params |
| Compression | zlib-9 | zstd-22 | ~-0.005 | Better ratio on int6 |
| MLP hidden | 1024 (2×) | 1536 (3×) | -0.01 | Budget freed by int6 |
| Train seq len | 1024 | 2048 | -0.01 | More context helps |
| Eval seq len | 1024 | 2048 | -0.005 | NTK-RoPE extrapolation |
| Matrix LR | 0.04 | 0.02 | -0.005 | Lower LR with int6 |
| Grad clip | 0 | 0.3 | ~-0.002 | Stabilizes training |
| Muon momentum | 0.95 | 0.99 | ~-0.002 | With warmup from 0.85-0.92 |
| Weight decay | 0 | 0.01-0.02 | ~-0.003 | Better generalization |
| Document isolation | No | Yes | -0.01 | Don't leak cross-doc context |

### C. FRONTIER TECHNIQUES (Novel, not yet combined with SOTA)
| Technique | Best BPB | Status | Potential |
|-----------|---------|--------|-----------|
| STE QAT (fake-int6 in training) | 1.1602 | In multiple PRs | HIGH — reduces quant gap to ~0.001 |
| NorMuon (row-wise RMS after NS) | 1.1602 | In PR #156 | MEDIUM — better per-step quality |
| OrthoInit + muP scaling | 1.1539 | In PR #135 | HIGH — faster convergence |
| BigramHash embedding | 1.1539 | In PR #135 | HIGH — token-pair info cheaply |
| SmearGate (prev-token blend) | 1.1539 | In PR #135 | MEDIUM — 512 params, helps embedding |
| SWA checkpoint averaging | 1.1666 | In PR #137 | LOW-MED — marginal at warmdown |
| BitNet b1.58 (ternary) | 1.2029 | In PR #139 | HIGH long-term — 64.5M params in 15MB |
| Depth Recurrence | 1.2196 | In PR #148 | MED — only 12.83MB artifact, room to grow |
| LoRA TTT (test-time train) | 1.1767 | In PR #152 | HIGH — orthogonal to all training improvements |
| Cross-layer param sharing + 4-bit QAT | ? | In PR #154 | UNKNOWN |

### D. UNTRIED COMBINATIONS (Our Opportunity)
| Combo | Expected BPB | Reasoning |
|-------|-------------|-----------|
| SOTA recipe + STE QAT | ~1.15 | Reduces quant gap on top of everything |
| SOTA + TTT | ~1.14-1.15 | TTT is orthogonal, uses only 1/10 eval budget |
| SOTA + Depth Recurrence | ~1.14-1.15 | 3×4 blocks = 12 effective layers in 9-block param budget |
| SOTA + BitNet hybrid | unknown | Replace some layers with ternary for more params |
| STE QAT + BigramHash + Depth Recurrence + TTT | ~1.13? | Kitchen sink — all orthogonal |

---

## Our Unique Angles

### 1. Depth Recurrence (Already Built!)
- Our implementation: `_get_block()` maps effective index to unique block with wrapping
- PR #148 shows 3×4 = 12 effective layers with only 3 unique blocks → 12.83MB artifact
- **Key insight**: 12.83MB is WAY under 16MB → room for MLP 3x, BigramHash, etc.
- **Gap**: PR #148 only got 1.2196 — didn't combine with int6, sliding window, QAT, etc.

### 2. Our Sweep Framework
- ML-guided predictor + 14 configs ready to go
- Deploy scripts for RunPod
- Can systematically find optimal recurrence config

### 3. What We Should Build
Stack ALL proven techniques onto depth recurrence:
- 3 blocks × 5 passes = 15 effective layers (only 3 unique blocks worth of params)
- Int6 + MLP 3x on the 3 unique blocks
- STE QAT during training
- FP16 tied embedding
- OrthoInit
- Sliding window eval
- Document-isolated eval
- BigramHash + SmearGate (if time permits)

---

## $25 RunPod Budget Plan

### Pricing
- 1×H100 SXM 80GB: ~$2.50/hr → 10 hours
- 8×H100 SXM 80GB: ~$20/hr → 75 minutes
- 1×A100 80GB: ~$1.50/hr → 16 hours (cheaper iteration)

### Experiment Sequence
| Phase | GPU | Time | Cost | Goal |
|-------|-----|------|------|------|
| 1. Baseline validation | 1×H100 | 20min | $0.83 | Reproduce SOTA ~1.175 on 1 GPU |
| 2. Depth recurrence sweep | 1×H100 | 2hr | $5.00 | Test 3×3, 3×4, 3×5, 5×3 configs |
| 3. Best config + int6/QAT | 1×H100 | 1hr | $2.50 | Add int6+STE QAT to best recurrence |
| 4. Full recipe on 8×H100 | 8×H100 | 30min | $10.00 | Validate final config at full scale |
| 5. TTT eval pass | 8×H100 | 20min | $6.67 | Add test-time training on top |
| **Total** | | | **$25.00** | |

### Key Rules
- ALWAYS save checkpoints to local disk
- Run validation at end of each experiment
- Log everything (ms/step, loss curves, artifact size)
- Kill pods immediately after each phase
