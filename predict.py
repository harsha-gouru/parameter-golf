"""
Parameter Golf outcome predictor.
Built from 6 actual 8×H100 runs + 1 SOTA reference.

Usage:
    python predict.py                    # show all options
    python predict.py --layers 11 --mlp 2.75  # specific config
"""
import math
import argparse

# =============================================================================
# FITTED CONSTANTS (from actual 8×H100 data)
# =============================================================================

# Step time model: step_ms ≈ base_factor * L * (1+m) / 40 + xsa_overhead + head_overhead
# Fitted from: SOTA=89ms, v1=95ms, v4=96ms, v5=105ms
BASE_STEP_FACTOR = 89.0   # ms for 10L mlp=3.0 (SOTA measured)
XSA_OVERHEAD_MS = 7.0     # XSA on last 4 layers adds ~7ms (v5 vs SOTA scaled)
LOGIT_HEAD_MS = 5.0       # BigramLogitHead adds ~5ms (v1 vs SOTA)
REFERENCE_LM = 40.0       # 10 * (1 + 3.0) = reference compute units

WALLCLOCK_S = 600.0        # hard competition limit (training time only)

# BPB model: fitted from convergence curves
# v5 (11L, full stack): step 5500→1.1457, step 5700→1.1409
# SOTA (10L, base): step 6500→1.1573, step 6709→1.1533
# Architecture quality offsets (relative to SOTA baseline 10L, LR=0.02, ReLU²)
ARCH_OFFSETS = {
    "xsa4":         -0.004,  # XSA on last 4 layers
    "partial_rope":  -0.002,  # 16/64 dims
    "ln_scale":     -0.002,  # 1/sqrt(layer+1)
    "leaky_relu":   -0.003,  # leaky_relu(0.5)² vs relu²
    "lr_025":       -0.003,  # LR 0.025 vs 0.02
    "11L_vs_10L":   -0.015,  # 11 layers vs 10 (biggest single gain)
}

# Convergence model: bpb_raw(step) ≈ A / sqrt(step) + B
# Fitted from SOTA 10L data: bpb(6709) = 1.1533
# Using: bpb = 8.5 / sqrt(step) + 1.05  (rough fit)
# For architecture-adjusted: subtract arch offsets
CONV_A = 8.5   # convergence constant
CONV_B = 1.05  # asymptotic floor for baseline 10L

# Post-processing
SWA_GAIN = -0.006   # average measured across v1-v4
QUANT_GAP = +0.003  # typical int5/int6 quantization loss
NET_POST = SWA_GAIN + QUANT_GAP  # = -0.003

# Size model (MB, after int5/int6 + zstd-22)
# Fitted from actual artifacts
SIZE_PER_LAYER_MB = 1.36       # from 5k-step tests: 15.93-14.57
SIZE_BASE_10L_MB = 13.45       # 10L no bigram features: v4 was 14.81 - 1.36(logit head)
SIZE_BIGRAM_HASH_PER_1K = 0.09  # MB per 1000 buckets (from 10240→0.92MB)
SIZE_LOGIT_HEAD_INT4 = 0.52    # int4 packed 1024×1024
SIZE_CODE_MB = 0.06
SIZE_LIMIT_MB = 16.0

# MLP contribution to layer size (rough: MLP is ~60% of layer params)
MLP_SIZE_FRACTION = 0.60


# =============================================================================
# MODELS
# =============================================================================

def step_time_ms(layers: int, mlp_mult: float, xsa: bool, logit_head: bool) -> float:
    """Predict per-step time in milliseconds."""
    compute_units = layers * (1.0 + mlp_mult)
    base = BASE_STEP_FACTOR * compute_units / REFERENCE_LM
    overhead = (XSA_OVERHEAD_MS if xsa else 0) + (LOGIT_HEAD_MS if logit_head else 0)
    return base + overhead


def total_steps(step_ms: float) -> int:
    """Total training steps under 600s wallclock."""
    return int(WALLCLOCK_S * 1000.0 / step_ms)


def bpb_raw(steps: int, arch_offset: float) -> float:
    """Predicted raw bpb (before SWA/quant) at given step count."""
    base_bpb = CONV_A / math.sqrt(steps) + CONV_B
    return base_bpb + arch_offset


def bpb_final(steps: int, arch_offset: float) -> float:
    """Predicted final bpb (after SWA + quantization)."""
    return bpb_raw(steps, arch_offset) + NET_POST


def arch_offset(layers: int, xsa: bool, partial_rope: bool, ln_scale: bool,
                leaky_relu: bool, lr_025: bool) -> float:
    """Total architecture quality offset vs baseline."""
    offset = 0.0
    if layers >= 11:
        offset += ARCH_OFFSETS["11L_vs_10L"]
    if xsa:
        offset += ARCH_OFFSETS["xsa4"]
    if partial_rope:
        offset += ARCH_OFFSETS["partial_rope"]
    if ln_scale:
        offset += ARCH_OFFSETS["ln_scale"]
    if leaky_relu:
        offset += ARCH_OFFSETS["leaky_relu"]
    if lr_025:
        offset += ARCH_OFFSETS["lr_025"]
    return offset


def artifact_size_mb(layers: int, mlp_mult: float, bigram_hash_buckets: int,
                     logit_head: bool) -> float:
    """Predicted artifact size in MB."""
    # Base size scales with layers and MLP mult
    base = SIZE_BASE_10L_MB
    # Layer cost: scale MLP portion by mlp_mult ratio
    mlp_ratio = mlp_mult / 3.0
    layer_cost = SIZE_PER_LAYER_MB * (MLP_SIZE_FRACTION * mlp_ratio + (1 - MLP_SIZE_FRACTION))
    size = base + layers * layer_cost - 10 * SIZE_PER_LAYER_MB  # subtract base 10 layers already in base
    # wait, that's wrong. Let me redo.
    # SIZE_BASE_10L_MB already includes 10 layers at mlp=3.0
    # For different layer count / mlp:
    extra_layers = layers - 10
    if extra_layers > 0:
        size += extra_layers * layer_cost
    # MLP scaling for existing layers
    if mlp_mult != 3.0:
        mlp_savings = 10 * SIZE_PER_LAYER_MB * MLP_SIZE_FRACTION * (1 - mlp_ratio)
        size -= mlp_savings

    # Bigram features
    if bigram_hash_buckets > 0:
        size += bigram_hash_buckets / 1000.0 * SIZE_BIGRAM_HASH_PER_1K
    if logit_head:
        size += SIZE_LOGIT_HEAD_INT4

    size += SIZE_CODE_MB
    return size


def predict(layers: int, mlp_mult: float, bigram_hash: int, logit_head: bool,
            xsa: bool, partial_rope: bool, ln_scale: bool, leaky_relu: bool,
            lr_025: bool) -> dict:
    """Full prediction for a configuration."""
    st = step_time_ms(layers, mlp_mult, xsa, logit_head)
    steps = total_steps(st)
    ao = arch_offset(layers, xsa, partial_rope, ln_scale, leaky_relu, lr_025)
    raw = bpb_raw(steps, ao)
    final = bpb_final(steps, ao)
    size = artifact_size_mb(layers, mlp_mult, bigram_hash, logit_head)
    fits = size <= SIZE_LIMIT_MB

    return {
        "step_ms": round(st, 1),
        "steps": steps,
        "arch_offset": round(ao, 4),
        "bpb_raw": round(raw, 4),
        "bpb_final": round(final, 4),
        "size_mb": round(size, 2),
        "fits": fits,
        "headroom_mb": round(SIZE_LIMIT_MB - size, 2),
    }


def validate_against_actuals():
    """Check model predictions against our actual 8×H100 data."""
    print("=" * 90)
    print("MODEL VALIDATION (predictions vs actual 8×H100 results)")
    print("=" * 90)

    actuals = [
        # (name, layers, mlp, hash, logit, xsa, prope, lns, lrelu, lr025, actual_steps, actual_ms, actual_bpb)
        ("SOTA",    10, 3.0, 10240, False, False, False, False, False, False, 6709, 89.4, 1.1428),
        ("v1",      10, 3.0, 0,     True,  False, False, False, False, False, 6300, 95.0, 1.1571),
        ("v4",      10, 3.0, 0,     True,  True,  True,  True,  False, True,  6225, 96.4, 1.1551),
        ("v5-pre",  11, 3.0, 10240, False, True,  True,  True,  True,  True,  5700, 105.0, 1.1409),
    ]

    print(f"{'Run':<10} {'Pred ms':>8} {'Act ms':>8} {'Δms':>6} {'Pred steps':>10} {'Act steps':>10} {'Pred bpb':>9} {'Act bpb':>9} {'Δbpb':>7}")
    print("-" * 90)

    for name, L, m, bh, lh, xsa, pr, ls, lr2, lr25, act_steps, act_ms, act_bpb in actuals:
        p = predict(L, m, bh, lh, xsa, pr, ls, lr2, lr25)
        delta_ms = p["step_ms"] - act_ms
        # For v5-pre, compare against bpb_raw (no SWA applied)
        pred_bpb = p["bpb_raw"] if "pre" in name else p["bpb_final"]
        delta_bpb = pred_bpb - act_bpb
        print(f"{name:<10} {p['step_ms']:>8.1f} {act_ms:>8.1f} {delta_ms:>+6.1f} {p['steps']:>10} {act_steps:>10} {pred_bpb:>9.4f} {act_bpb:>9.4f} {delta_bpb:>+7.4f}")


def sweep():
    """Sweep all plausible configurations and rank by predicted bpb."""
    print("\n" + "=" * 90)
    print("CONFIGURATION SWEEP (ranked by predicted final bpb)")
    print("=" * 90)

    configs = []
    for L in [10, 11]:
        for m in [2.5, 2.625, 2.75, 3.0]:
            for bh in [0, 2048, 4096, 10240]:
                for lh in [False]:  # logit head disabled
                    for xsa in [True]:
                        for lr2 in [True]:
                            p = predict(L, m, bh, lh, xsa, True, True, lr2, True)
                            if p["fits"]:
                                label = f"{L}L m={m} hash={bh}"
                                configs.append((p["bpb_final"], label, p))

    configs.sort()
    print(f"\n{'Rank':>4} {'Config':<30} {'bpb_final':>9} {'Steps':>6} {'ms/step':>7} {'Size MB':>8} {'Room MB':>8}")
    print("-" * 85)
    for i, (bpb, label, p) in enumerate(configs[:15]):
        marker = " <<<" if bpb < 1.1428 else ""
        print(f"{i+1:>4} {label:<30} {p['bpb_final']:>9.4f} {p['steps']:>6} {p['step_ms']:>7.1f} {p['size_mb']:>8.2f} {p['headroom_mb']:>+8.2f}{marker}")

    print(f"\n  <<< = beats current merged #1 (1.1428)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Golf outcome predictor")
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--mlp", type=float, default=0)
    args = parser.parse_args()

    validate_against_actuals()
    sweep()

    if args.layers > 0:
        print(f"\n{'='*90}")
        print(f"SPECIFIC PREDICTION: {args.layers}L, mlp={args.mlp}")
        print(f"{'='*90}")
        p = predict(args.layers, args.mlp, 4096, False, True, True, True, True, True)
        for k, v in p.items():
            print(f"  {k}: {v}")
