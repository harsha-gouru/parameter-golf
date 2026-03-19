#!/usr/bin/env python3
"""
Automated sweep runner for Parameter Golf.
Runs round 1 experiments, then uses ML to generate and run round 2.

Usage:
    python experiments/run_sweep.py                    # full auto sweep
    python experiments/run_sweep.py --rounds 1         # just round 1
    python experiments/run_sweep.py --nproc 8          # 8 GPUs
    python experiments/run_sweep.py --short            # quick runs (2 min each)
"""
import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from runner import default_config, run_experiment, estimate_model_size_mb, load_results, print_leaderboard
from predictor import suggest_configs, analyze

CONFIGS_DIR = Path(__file__).parent / "configs"


def run_round1(nproc=1, short=False):
    """Run all round 1 configs."""
    sweep_dir = CONFIGS_DIR / "sweep_round1"
    configs = sorted(sweep_dir.glob("*.json"))
    print(f"\n{'='*60}")
    print(f"ROUND 1: {len(configs)} experiments")
    print(f"{'='*60}\n")

    # Check which have already been run
    existing = {r["run_id"] for r in load_results()}

    for cfg_path in configs:
        with open(cfg_path) as f:
            overrides = json.load(f)

        name = overrides.get("_name", cfg_path.stem)
        run_id = f"r1_{name}"

        if run_id in existing:
            print(f"[SKIP] {run_id} already run")
            continue

        config = default_config()
        # Handle special configs
        clean_overrides = {k: v for k, v in overrides.items() if not k.startswith("_")}
        config.update(clean_overrides)

        if short:
            config["MAX_WALLCLOCK_SECONDS"] = 120.0
            config["ITERATIONS"] = 2000
            config["VAL_LOSS_EVERY"] = 500

        # Handle vocab-specific dataset paths
        vocab = config.get("VOCAB_SIZE", 1024)
        if vocab != 1024:
            variant = f"sp{vocab}"
            config["DATA_PATH"] = f"./data/datasets/fineweb10B_{variant}"
            config["TOKENIZER_PATH"] = f"./data/tokenizers/fineweb_{vocab}_bpe.model"

        # Determine script
        script = "train_gpt.py"

        print(f"\n--- Running: {run_id} ---")
        result = run_experiment(config, run_id=run_id, script=script, nproc=nproc)

        if result["status"] != "success":
            print(f"[WARN] {run_id} failed: exit_code={result.get('exit_code')}")
            if result.get("error_tail"):
                print(result["error_tail"][-500:])


def run_round2(nproc=1, n_experiments=5, short=False):
    """Use ML predictor to generate and run round 2 configs."""
    print(f"\n{'='*60}")
    print(f"ROUND 2: ML-suggested experiments")
    print(f"{'='*60}\n")

    suggestions = suggest_configs(n=n_experiments)
    if not suggestions:
        print("No suggestions available. Need more round 1 data.")
        return

    # Save suggested configs for reproducibility
    r2_dir = CONFIGS_DIR / "sweep_round2_auto"
    r2_dir.mkdir(parents=True, exist_ok=True)

    existing = {r["run_id"] for r in load_results()}

    for i, (config, predicted_bpb) in enumerate(suggestions, 1):
        run_id = f"r2_ml_{i:02d}"

        if run_id in existing:
            print(f"[SKIP] {run_id} already run")
            continue

        # Save config
        diff = {k: v for k, v in config.items()
                if v != default_config().get(k) and not k.startswith("_")}
        diff["_predicted_bpb"] = predicted_bpb
        cfg_path = r2_dir / f"{run_id}.json"
        with open(cfg_path, "w") as f:
            json.dump(diff, f, indent=2)

        if short:
            config["MAX_WALLCLOCK_SECONDS"] = 120.0
            config["ITERATIONS"] = 2000

        # Handle vocab paths
        vocab = config.get("VOCAB_SIZE", 1024)
        if vocab != 1024:
            variant = f"sp{vocab}"
            config["DATA_PATH"] = f"./data/datasets/fineweb10B_{variant}"
            config["TOKENIZER_PATH"] = f"./data/tokenizers/fineweb_{vocab}_bpe.model"

        print(f"\n--- Running: {run_id} (predicted BPB: {predicted_bpb:.4f}) ---")
        print(f"    Changes: {json.dumps({k:v for k,v in diff.items() if not k.startswith('_')})}")
        result = run_experiment(config, run_id=run_id, script="train_gpt.py", nproc=nproc)

        if result["status"] != "success":
            print(f"[WARN] {run_id} failed")


def run_round3(nproc=1, n_experiments=5, short=False):
    """Round 3: tighter mutations around the best result."""
    print(f"\n{'='*60}")
    print(f"ROUND 3: Fine-tuning best config")
    print(f"{'='*60}\n")

    results = load_results()
    successful = [r for r in results if r.get("val_bpb") is not None]
    if not successful:
        print("No successful runs yet.")
        return

    best = min(successful, key=lambda r: r["val_bpb"])
    print(f"Best so far: {best['run_id']} with BPB={best['val_bpb']:.4f}")

    # Generate tight mutations
    from predictor import mutate_config
    existing = {r["run_id"] for r in results}

    count = 0
    for i in range(n_experiments * 5):  # oversample then filter
        if count >= n_experiments:
            break

        run_id = f"r3_fine_{count+1:02d}"
        if run_id in existing:
            count += 1
            continue

        config = mutate_config(best["config"], mutation_rate=0.15)
        if estimate_model_size_mb(config) > 15.5:
            continue

        if short:
            config["MAX_WALLCLOCK_SECONDS"] = 120.0
            config["ITERATIONS"] = 2000

        vocab = config.get("VOCAB_SIZE", 1024)
        if vocab != 1024:
            config["DATA_PATH"] = f"./data/datasets/fineweb10B_sp{vocab}"
            config["TOKENIZER_PATH"] = f"./data/tokenizers/fineweb_{vocab}_bpe.model"

        print(f"\n--- Running: {run_id} ---")
        result = run_experiment(config, run_id=run_id, script="train_gpt.py", nproc=nproc)
        count += 1


def main():
    parser = argparse.ArgumentParser(description="Automated Parameter Golf Sweep")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds (1-3)")
    parser.add_argument("--nproc", type=int, default=1, help="GPUs per run")
    parser.add_argument("--short", action="store_true", help="Short runs (2 min each)")
    parser.add_argument("--r2-count", type=int, default=5, help="Experiments per ML round")
    args = parser.parse_args()

    t0 = datetime.now()

    if args.rounds >= 1:
        run_round1(nproc=args.nproc, short=args.short)
        print_leaderboard()

    if args.rounds >= 2:
        analyze()
        run_round2(nproc=args.nproc, n_experiments=args.r2_count, short=args.short)
        print_leaderboard()

    if args.rounds >= 3:
        analyze()
        run_round3(nproc=args.nproc, n_experiments=args.r2_count, short=args.short)
        print_leaderboard()

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nTotal sweep time: {elapsed/60:.1f} minutes")
    print("\nFinal analysis:")
    analyze()


if __name__ == "__main__":
    main()
