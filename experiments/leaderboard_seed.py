#!/usr/bin/env python3
"""
Leaderboard seed data for Parameter Golf predictor.
Encodes known upstream submissions + scores as synthetic training data.

Usage:
    python experiments/leaderboard_seed.py          # print all entries
    python experiments/leaderboard_seed.py --json    # output as JSONL
"""
import json
import sys
from datetime import datetime


def upstream_baseline_config():
    """Upstream train_gpt.py defaults at time of leaderboard submissions.
    These differ from runner.py's default_config() in some values."""
    return {
        "NUM_LAYERS": 9,
        "MODEL_DIM": 512,
        "NUM_HEADS": 8,
        "NUM_KV_HEADS": 4,
        "MLP_MULT": 2,
        "VOCAB_SIZE": 1024,
        "TIE_EMBEDDINGS": 1,
        "ROPE_BASE": 10000.0,
        "LOGIT_SOFTCAP": 30.0,
        "QK_GAIN_INIT": 1.5,
        "TIED_EMBED_INIT_STD": 0.005,
        "TRAIN_SEQ_LEN": 1024,
        "TRAIN_BATCH_TOKENS": 524288,
        "ITERATIONS": 20000,
        "WARMDOWN_ITERS": 1200,
        "WARMUP_STEPS": 20,
        "MAX_WALLCLOCK_SECONDS": 600.0,
        "TIED_EMBED_LR": 0.05,
        "MATRIX_LR": 0.04,
        "SCALAR_LR": 0.04,
        "MUON_MOMENTUM": 0.95,
        "MUON_BACKEND_STEPS": 5,
        "MUON_MOMENTUM_WARMUP_START": 0.85,
        "MUON_MOMENTUM_WARMUP_STEPS": 500,
        "BETA1": 0.9,
        "BETA2": 0.95,
        "ADAM_EPS": 1e-8,
        "GRAD_CLIP_NORM": 0.0,
        "QAT_ENABLED": 1,
        "QAT_START_FRAC": 0.75,
        "QAT_RAMP_FRAC": 0.05,
        "COSINE_WARMDOWN": 1,
        "DEPTH_RECURRENCE": 0,
        "WEIGHT_TYING_LAYERS": 0,
        # New keys (defaults for submissions that don't override them)
        "FP16_EMBED": 1,
        "EVAL_STRIDE": 64,
        "QUANT_BITS": 6,
        "WEIGHT_DECAY": 0.01,
        "MUON_WEIGHT_DECAY": 0.02,
    }


# (run_id, val_bpb, overrides_dict)
LEADERBOARD_ENTRIES = [
    ("lb_baseline", 1.2244, {}),
    ("lb_lower_lr", 1.2230, {
        "MATRIX_LR": 0.02, "SCALAR_LR": 0.02, "TIED_EMBED_LR": 0.03,
    }),
    ("lb_fp16_embed", 1.2197, {
        "FP16_EMBED": 1, "WARMDOWN_ITERS": 3600, "MATRIX_LR": 0.06,
    }),
    ("lb_10l_mixed", 1.2147, {
        "NUM_LAYERS": 10, "MATRIX_LR": 0.02,
        # QUANT_BITS=6 is already the upstream default, no override needed
    }),
    ("lb_seq2048", 1.2058, {
        "TRAIN_SEQ_LEN": 2048, "MATRIX_LR": 0.032,
    }),
    ("lb_seq4096", 1.2014, {
        "TRAIN_SEQ_LEN": 4096, "TRAIN_BATCH_TOKENS": 393216,
        "MUON_MOMENTUM": 0.99, "WARMDOWN_ITERS": 3000,
    }),
    ("lb_lora_ttt", 1.1928, {
        # Eval-only trick with baseline training
    }),
    ("lb_sliding_eval", 1.1925, {
        "EVAL_STRIDE": 64,
    }),
    ("lb_sliding_10l", 1.1748, {
        "NUM_LAYERS": 10, "WARMDOWN_ITERS": 2500,
        "TIED_EMBED_LR": 0.10, "EVAL_STRIDE": 64, "FP16_EMBED": 1,
    }),
    ("lb_warmdown_quant", 1.1574, {
        "WARMDOWN_ITERS": 20000, "MATRIX_LR": 0.06,
        "GRAD_CLIP_NORM": 1.0, "FP16_EMBED": 1,
    }),
]


def get_leaderboard_data():
    """Return leaderboard entries as results.jsonl-compatible dicts."""
    base = upstream_baseline_config()
    results = []
    for run_id, val_bpb, overrides in LEADERBOARD_ENTRIES:
        config = dict(base)
        config.update(overrides)
        results.append({
            "run_id": run_id,
            "config": config,
            "val_bpb": val_bpb,
            "status": "leaderboard",
            "source": "upstream_leaderboard",
            "timestamp": "2026-03-19T00:00:00",
        })
    return results


def main():
    data = get_leaderboard_data()
    if "--json" in sys.argv:
        for entry in data:
            print(json.dumps(entry))
    else:
        print(f"Leaderboard seed: {len(data)} entries\n")
        print(f"{'Run ID':<25} {'BPB':<10} Key overrides")
        print(f"{'-'*70}")
        for entry in data:
            base = upstream_baseline_config()
            diff = {k: v for k, v in entry["config"].items() if v != base.get(k)}
            print(f"{entry['run_id']:<25} {entry['val_bpb']:<10.4f} {diff}")


if __name__ == "__main__":
    main()
