#!/usr/bin/env python3
"""
Experiment runner for Parameter Golf.
Runs configs, logs results, feeds data to the predictor.

Usage:
    # Run a single experiment
    python experiments/runner.py --config experiments/configs/baseline.json

    # Run a sweep (all configs in a directory)
    python experiments/runner.py --sweep experiments/configs/sweep_depth_recurrence/

    # Run the predictor to suggest next config
    python experiments/runner.py --suggest --n 5
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

EXPERIMENTS_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENTS_DIR / "results.jsonl"
REPO_ROOT = EXPERIMENTS_DIR.parent


def default_config():
    """Baseline config — all the knobs we can turn."""
    return {
        # Architecture
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
        # Training
        "TRAIN_SEQ_LEN": 1024,
        "TRAIN_BATCH_TOKENS": 524288,
        "ITERATIONS": 20000,
        "WARMDOWN_ITERS": 1200,
        "WARMUP_STEPS": 20,
        "MAX_WALLCLOCK_SECONDS": 600.0,
        # Optimizer
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
        # QAT
        "QAT_ENABLED": 1,
        "QAT_START_FRAC": 0.75,
        "QAT_RAMP_FRAC": 0.05,
        # Cosine warmdown
        "COSINE_WARMDOWN": 1,
        # Custom architecture flags (for our mods)
        "DEPTH_RECURRENCE": 0,  # number of times to loop blocks (0=off)
        "WEIGHT_TYING_LAYERS": 0,  # tie layer weights (0=off)
    }


def estimate_params(config):
    """Rough parameter count estimate without building the model."""
    d = config["MODEL_DIM"]
    v = config["VOCAB_SIZE"]
    n = config["NUM_LAYERS"]
    kv = config["NUM_KV_HEADS"]
    hd = d // config["NUM_HEADS"]
    mlp_h = d * config["MLP_MULT"]

    embed = v * d if config["TIE_EMBEDDINGS"] else 2 * v * d
    per_block = (
        d * d  # c_q
        + d * (kv * hd)  # c_k
        + d * (kv * hd)  # c_v
        + d * d  # proj
        + d * mlp_h  # fc
        + mlp_h * d  # mlp proj
        + d * 3  # scales + q_gain heads
        + d * 2  # resid_mix
    )
    unique_blocks = n  # NUM_LAYERS = number of unique block modules
    recurrence = max(config.get("DEPTH_RECURRENCE", 1), 1)
    effective_layers = n * recurrence
    # skip_weights are based on effective layers (encoder/decoder split)
    num_encoder = effective_layers // 2
    num_decoder = effective_layers - num_encoder
    num_skip = min(num_encoder, num_decoder)
    skip_weights = num_skip * d
    total = embed + unique_blocks * per_block + skip_weights
    return total


def estimate_model_size_mb(config):
    """Estimate compressed model size in MB (int8 + zlib ~0.85x)."""
    params = estimate_params(config)
    # int8 = 1 byte/param + scales (~0.5% overhead), zlib ~85% of that
    return (params * 1.005 * 0.85) / 1_000_000


def run_experiment(config, run_id=None, script="train_gpt.py", nproc=1, dry_run=False):
    """Run a single training experiment and parse the results."""
    if run_id is None:
        run_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Estimate and check size constraint
    est_size = estimate_model_size_mb(config)
    est_params = estimate_params(config)
    print(f"[{run_id}] Estimated params: {est_params:,} | Size: {est_size:.2f}MB", flush=True)
    if est_size > 16.0:
        print(f"[{run_id}] WARNING: Estimated size {est_size:.2f}MB exceeds 16MB limit!", flush=True)

    if dry_run:
        result = {
            "run_id": run_id,
            "config": config,
            "estimated_params": est_params,
            "estimated_size_mb": est_size,
            "status": "dry_run",
            "timestamp": datetime.now().isoformat(),
        }
        save_result(result)
        return result

    # Build env
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    for k, v in config.items():
        if k.startswith("_") or k in ("WEIGHT_TYING_LAYERS",):
            continue
        env[k] = str(v)

    # Pick launch command
    if nproc > 1:
        cmd = [
            "torchrun", "--standalone", f"--nproc_per_node={nproc}",
            str(REPO_ROOT / script),
        ]
    else:
        cmd = [sys.executable, str(REPO_ROOT / script)]

    print(f"[{run_id}] Running: {' '.join(cmd)}", flush=True)
    t0 = time.time()

    proc = subprocess.run(
        cmd, env=env, cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=7200,  # 2hr max
    )
    elapsed = time.time() - t0

    # Parse output for metrics
    output = proc.stdout + "\n" + proc.stderr
    val_bpb = None
    val_loss = None
    model_size = None

    for line in output.split("\n"):
        if "final_int8_zlib_roundtrip_exact" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    val_bpb = float(part.split(":")[1])
                elif part.startswith("val_loss:"):
                    val_loss = float(part.split(":")[1])
        if "Total submission size int8+zlib:" in line:
            try:
                model_size = int(line.split(":")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if "Serialized model int8+zlib:" in line:
            try:
                model_size = int(line.split(":")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass

    result = {
        "run_id": run_id,
        "config": config,
        "val_bpb": val_bpb,
        "val_loss": val_loss,
        "model_size_bytes": model_size,
        "estimated_params": est_params,
        "estimated_size_mb": est_size,
        "elapsed_seconds": elapsed,
        "exit_code": proc.returncode,
        "status": "success" if proc.returncode == 0 and val_bpb is not None else "failed",
        "timestamp": datetime.now().isoformat(),
        "nproc": nproc,
        "script": script,
    }

    if proc.returncode != 0:
        # Save last 50 lines of output for debugging
        result["error_tail"] = "\n".join(output.strip().split("\n")[-50:])

    save_result(result)
    print(f"[{run_id}] Done in {elapsed:.0f}s | BPB: {val_bpb} | Loss: {val_loss} | Size: {model_size}", flush=True)
    return result


def save_result(result):
    """Append result to JSONL file."""
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")


def load_results():
    """Load all experiment results."""
    if not RESULTS_FILE.exists():
        return []
    results = []
    for line in RESULTS_FILE.read_text().strip().split("\n"):
        if line:
            results.append(json.loads(line))
    return results


def print_leaderboard():
    """Print results sorted by BPB."""
    results = load_results()
    successful = [r for r in results if r.get("val_bpb") is not None]
    successful.sort(key=lambda r: r["val_bpb"])
    print(f"\n{'='*80}")
    print(f"{'Rank':<5} {'Run ID':<30} {'BPB':<10} {'Loss':<10} {'Size(MB)':<10} {'Time(s)':<10}")
    print(f"{'='*80}")
    for i, r in enumerate(successful, 1):
        size_mb = r.get("model_size_bytes", 0) or 0
        size_mb = size_mb / 1_000_000
        print(f"{i:<5} {r['run_id']:<30} {r['val_bpb']:<10.4f} {r.get('val_loss', 0):<10.4f} {size_mb:<10.2f} {r.get('elapsed_seconds', 0):<10.0f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Parameter Golf Experiment Runner")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--sweep", type=str, help="Directory of config JSONs to sweep")
    parser.add_argument("--suggest", action="store_true", help="Run predictor to suggest next config")
    parser.add_argument("--n", type=int, default=5, help="Number of suggestions")
    parser.add_argument("--leaderboard", action="store_true", help="Print experiment leaderboard")
    parser.add_argument("--dry-run", action="store_true", help="Estimate only, don't train")
    parser.add_argument("--nproc", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--script", type=str, default="train_gpt.py", help="Training script")
    parser.add_argument("--run-id", type=str, help="Custom run ID")
    args = parser.parse_args()

    if args.leaderboard:
        print_leaderboard()
        return

    if args.suggest:
        from predictor import suggest_configs
        suggestions = suggest_configs(n=args.n)
        for i, (cfg, score) in enumerate(suggestions, 1):
            print(f"\nSuggestion {i} (predicted BPB: {score:.4f}):")
            diff = {k: v for k, v in cfg.items() if v != default_config().get(k)}
            print(f"  Changes from baseline: {json.dumps(diff, indent=2)}")
            print(f"  Est. params: {estimate_params(cfg):,} | Est. size: {estimate_model_size_mb(cfg):.2f}MB")
        return

    if args.config:
        with open(args.config) as f:
            overrides = json.load(f)
        config = default_config()
        config.update(overrides)
        run_experiment(config, run_id=args.run_id, script=args.script,
                      nproc=args.nproc, dry_run=args.dry_run)
        return

    if args.sweep:
        sweep_dir = Path(args.sweep)
        configs = sorted(sweep_dir.glob("*.json"))
        print(f"Running sweep of {len(configs)} configs from {sweep_dir}")
        for cfg_path in configs:
            with open(cfg_path) as f:
                overrides = json.load(f)
            config = default_config()
            config.update(overrides)
            run_id = args.run_id or cfg_path.stem
            run_experiment(config, run_id=run_id, script=args.script,
                          nproc=args.nproc, dry_run=args.dry_run)

    print_leaderboard()


if __name__ == "__main__":
    main()
