# Parameter Golf — Final Strategy
*Compiled: 2026-03-19 from 4 parallel research agents + open PR analysis*

## TL;DR

**Goal**: Sub-1.15 BPB submission. Current best open PR is 1.1539.

**Our edge**: Depth recurrence + everything else. Nobody has combined depth recurrence
with the full SOTA recipe (int6, MLP3x, STE QAT, sliding window, OrthoInit, TTT).
The depth recurrence PR #148 scored only 1.2196 because it used NONE of those.

**Budget**: $25 RunPod credit = ~75 min on 8×H100 or ~10hr on 1×H100.

---

## What We Now Know (Agent Research Synthesis)

### The Scoring Formula
```
final_bpb = pre_quant_bpb + quant_gap + eval_penalty
```
- **pre_quant_bpb**: How good the model actually is (training quality)
- **quant_gap**: BPB lost to int8/int6 compression (typically 0.001–0.014)
- **eval_penalty**: BPB lost from suboptimal eval (sliding window recovers ~0.03)

Every top entry attacks ALL THREE simultaneously.

### The 6 Pillars (Table Stakes for Any Competitive Entry)
| # | Pillar | BPB Impact | Implementation |
|---|--------|-----------|----------------|
| 1 | Int6 quantization | -0.01 (frees bytes for MLP3x) | `max_val=31` instead of 127 |
| 2 | MLP 3× (hidden=1536) | -0.01 | Reinvest int6 savings |
| 3 | STE QAT | -0.005 (kills quant gap) | Fake-quantize in forward, STE in backward |
| 4 | Sliding window eval stride=64 | -0.03 | `forward_logits` + strided scoring |
| 5 | FP16 tied embedding | -0.005 | Skip int8 for `tok_emb.weight` |
| 6 | OrthoInit + lower LR (0.02) | -0.005 | Orthogonal init, muP scaling |

### Proven Knobs (Exact Values from Top Submissions)
| Knob | Best Value | Source |
|------|-----------|--------|
| matrix_lr | 0.02 | PR #135, #137 |
| scalar_lr | 0.02 | PR #135, #137 |
| tied_embed_lr | 0.03 | PR #135, #137 |
| warmdown_iters | 3000 | PR #135, #137 |
| grad_clip_norm | 0.3 | PR #135, #137 |
| muon_momentum | 0.99 | PR #135, #137 |
| muon_momentum_warmup_start | 0.92 | PR #137 |
| muon_momentum_warmup_steps | 1500 | PR #137 |
| weight_decay (AdamW) | 0.01 | SOTA merged |
| muon_weight_decay | 0.02 | SOTA merged |
| train_seq_len | 2048 | PR #135 |
| train_batch_tokens | 786432 | PR #135 |
| eval_stride | 64 | All top entries |
| compression | zstd-22 | PR #135 (better than zlib-9) |
| logit_softcap | 30.0 | Default (unchanged) |

### What Makes PR #135 (1.1539) Win
1. **OrthoInit**: All large matrices get orthogonal init + `1/sqrt(2*num_layers)` muP output scaling
2. **BigramHash**: 4096-bucket hash table (dim=128→512) injecting token-pair bigram info
3. **SmearGate**: Learned gate blending token embedding with previous token's (~512 params)
4. **zstd-22**: Better compression than zlib-9 on int6 data
5. Everything else from the 6 pillars above

### Depth Recurrence — The Untapped Multiplier
PR #148 proved the concept: **3 blocks × 4 passes = 12 effective layers, only 12.83MB artifact**.
That's 3.17MB of headroom — enough for MLP 3×, BigramHash, and wider dim.

Key findings from our agent analysis:
- Our `_get_block(i % num_unique_layers)` implementation is correct and clean
- U-Net skip connections span the full effective depth (gradient flows through all passes)
- **Compatibility issues to solve**:
  - Int6: With block reuse, same block appears at multiple positions. Apply int6 to ALL unique blocks (simpler than selective).
  - TTT/LoRA: Current upstream TTT uses `self.blocks[i]` directly → need to adapt for `_get_block()` wrapping. LoRA adapters need to be indexed by effective layer, not unique block.

### Why Depth Recurrence + Full Recipe Could Win
| Config | Unique Blocks | MLP | Effective Layers | Est. Artifact | Expected BPB |
|--------|--------------|-----|-----------------|---------------|-------------|
| Current SOTA (#135) | 9 | 3× | 9 | 15.2MB | 1.1539 |
| Ours: 3×5 wide | 3 | 3× | 15 | ~13MB | **~1.13-1.14?** |
| Ours: 4×4 | 4 | 3× | 16 | ~15MB | **~1.12-1.14?** |
| Ours + TTT | 3×5 | 3× | 15+TTT | ~13MB | **~1.11-1.13?** |

The logic: more effective depth with fewer params → either wider model OR more MLP capacity
within 16MB. 15 effective layers from 3 unique blocks is genuinely novel.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Depth recurrence hurts convergence speed (fewer unique params, more passes = slower ms/step) | HIGH | PR #148 was 133ms/step vs 43ms baseline. Budget: fewer total steps. Must compensate with better per-step quality. |
| 3×5 is too slow for 600s wallclock | MEDIUM | Fewer unique blocks → smaller matmuls → may not be 5× slower. Test on 1×H100 first. |
| Block reuse creates gradient conflicts (same weights optimized for multiple positions) | MEDIUM | The U-Net skip connections + resid_mix should help. PR #148 proved it works for 3×4. |
| $25 not enough for sufficient experimentation | HIGH | Use 1×H100 for all iteration. Only use 8×H100 for final 1-2 validation runs. |

---

## Concrete Experiment Plan ($25 Budget)

### Pre-Flight (Local, Free)
- [x] Rebase onto upstream/main ✓
- [ ] Verify train_gpt.py compiles and runs on MLX (smoke test, 50 iterations)
- [ ] Estimate artifact sizes for target configs (param count × compression ratio)
- [ ] Pre-compute expected ms/step from param counts

### Phase 1: Reproduce & Baseline ($2.50, 1×H100, 1hr)
Run the current SOTA recipe (PR #135 style) on 1 GPU to establish our baseline:
```bash
NUM_LAYERS=9 MLP_HIDDEN=1536 MATRIX_LR=0.02 SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3 \
MUON_MOMENTUM=0.99 TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 \
EVAL_STRIDE=64 MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```
Expected: ~1.18-1.20 on 1 GPU (fewer steps than 8×H100).

### Phase 2: Depth Recurrence Sweep ($5.00, 1×H100, 2hr)
Test 4 configs with ALL proven techniques applied:
```
A: 3 blocks × 3 passes (9 eff), dim=768, MLP 3×
B: 3 blocks × 5 passes (15 eff), dim=768, MLP 3×
C: 4 blocks × 4 passes (16 eff), dim=640, MLP 3×
D: 3 blocks × 5 passes (15 eff), dim=896, MLP 3× (max budget)
```
Each run: 10min + eval. Compare BPB, ms/step, artifact size.

### Phase 3: Int6 + STE QAT Integration ($2.50, 1×H100, 1hr)
Take best config from Phase 2, add:
- Int6 quantization (all unique blocks)
- STE fake-int6 QAT in forward pass
- zstd-22 compression
- FP16 tied embedding
Verify artifact ≤16MB and measure BPB improvement.

### Phase 4: Full Validation ($10.00, 8×H100, 30min)
Run best config on 8×H100 with full 600s wallclock:
```bash
DEPTH_RECURRENCE=5 NUM_LAYERS=3 MODEL_DIM=768 MLP_HIDDEN=2304 \
# ... all proven knobs ...
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
3 seed runs for p<0.01 statistical significance.

### Phase 5: TTT Layer ($5.00, 8×H100, 15min)
If Phase 4 is strong, add LoRA TTT on top:
- Adapt TTT loop for `_get_block()` indexing
- LoRA rank 8, targets c_q + c_v + lm_head
- Document-isolated, chunk=256
Expected: additional -0.003 to -0.01 BPB

### Contingency: If Depth Recurrence Disappoints
Fall back to: reproduce PR #135 recipe (OrthoInit + Int6 + MLP3x + BigramHash + SmearGate)
and add TTT on top. That alone could reach ~1.14.

---

## Implementation Checklist

### Must Implement (Before Any GPU Run)
- [ ] Int6 quantization in `quantize_state_dict_int8` (change max_val to 31)
- [ ] zstd-22 compression option (replace zlib with zstd)
- [ ] STE QAT: fake-int6 quantize in forward, straight-through gradient in backward
- [ ] OrthoInit: orthogonal init for all 2D weights, muP output scaling
- [ ] Sliding window eval with `forward_logits` and stride=64
- [ ] FP16 tied embedding passthrough in quantization
- [ ] NTK-aware RoPE scaling for eval at longer context
- [ ] Document-isolated eval (BOS boundary detection)

### Should Implement (High Value)
- [ ] BigramHash embedding (4096 buckets, dim=128→512)
- [ ] SmearGate (previous-token embedding blend)
- [ ] NorMuon (row-wise RMS after Newton-Schulz)
- [ ] SWA checkpoint averaging during warmdown

### Nice to Have (If Time)
- [ ] LoRA TTT adaptation for `_get_block()` indexing
- [ ] Phase-transition resid_mix init
- [ ] Overtone spectral embedding init

---

## Key Numbers to Remember
- 16MB = 16,000,000 bytes (decimal, NOT 16 MiB)
- Baseline 9L×512: ~17.1M params, ~15.9MB int8+zlib
- 10L×512: ~18.9M params, ~17.6MB int8 (over budget — needs int6)
- Int6 saves ~25% vs int8 in compressed size
- zstd-22 saves ~5-10% vs zlib-9
- Sliding window stride=64 adds ~160s eval time (within 10min budget)
- 1 BPB improvement of 0.005 = minimum for SOTA record
- p<0.01 required = need 3+ seed runs
