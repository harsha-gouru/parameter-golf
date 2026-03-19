#!/usr/bin/env python3
"""
Bayesian hyperparameter optimizer for Parameter Golf.
Uses experiment results to predict which configs will give the best BPB.

Uses a Random Forest (lightweight, no extra deps beyond sklearn) to model
config → BPB, then samples candidates and picks the most promising ones.

Usage:
    python experiments/predictor.py --suggest 5
    python experiments/predictor.py --analyze
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

EXPERIMENTS_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENTS_DIR / "results.jsonl"

# Search space definition — (min, max, step) for each tunable param
SEARCH_SPACE = {
    # Architecture
    "NUM_LAYERS": (3, 18, 1),
    "MODEL_DIM": (128, 1024, 64),
    "NUM_HEADS": (2, 16, 2),
    "NUM_KV_HEADS": (1, 8, 1),
    "MLP_MULT": (1, 4, 1),
    "DEPTH_RECURRENCE": (1, 6, 1),
    "ROPE_BASE": (1000.0, 100000.0, 1000.0),
    "LOGIT_SOFTCAP": (10.0, 50.0, 5.0),
    "QK_GAIN_INIT": (0.5, 3.0, 0.25),
    # Training
    "TRAIN_SEQ_LEN": (256, 2048, 256),
    "WARMDOWN_ITERS": (200, 3000, 200),
    # Optimizer
    "TIED_EMBED_LR": (0.005, 0.2, 0.005),
    "MATRIX_LR": (0.005, 0.15, 0.005),
    "SCALAR_LR": (0.005, 0.15, 0.005),
    "MUON_MOMENTUM": (0.85, 0.99, 0.01),
    "MUON_BACKEND_STEPS": (3, 10, 1),
    # QAT
    "QAT_START_FRAC": (0.5, 0.9, 0.05),
    "QAT_RAMP_FRAC": (0.02, 0.15, 0.01),
}

# Feature names in consistent order
FEATURE_NAMES = sorted(SEARCH_SPACE.keys())


def load_results():
    """Load successful experiment results."""
    if not RESULTS_FILE.exists():
        return []
    results = []
    for line in RESULTS_FILE.read_text().strip().split("\n"):
        if line:
            r = json.loads(line)
            if r.get("val_bpb") is not None and r.get("status") == "success":
                results.append(r)
    return results


def config_to_features(config):
    """Convert config dict to feature vector."""
    return np.array([float(config.get(k, 0)) for k in FEATURE_NAMES])


def estimate_model_size_mb(config):
    """Quick size estimate."""
    d = config.get("MODEL_DIM", 512)
    v = config.get("VOCAB_SIZE", 1024)
    n = config.get("NUM_LAYERS", 9)
    kv = config.get("NUM_KV_HEADS", 4)
    hd = d // max(config.get("NUM_HEADS", 8), 1)
    mlp_h = d * config.get("MLP_MULT", 2)
    per_block = d * d + d * kv * hd * 2 + d * d + d * mlp_h * 2 + d * 5
    total = v * d + n * per_block + (n // 2) * d
    return (total * 1.005 * 0.85) / 1_000_000


def random_config():
    """Sample a random config from the search space."""
    from runner import default_config
    config = default_config()
    for k, (lo, hi, step) in SEARCH_SPACE.items():
        n_steps = int((hi - lo) / step)
        val = lo + random.randint(0, n_steps) * step
        if isinstance(config.get(k), int):
            val = int(val)
        config[k] = val
    # Constraints
    if config["NUM_KV_HEADS"] > config["NUM_HEADS"]:
        config["NUM_KV_HEADS"] = config["NUM_HEADS"]
    if config["NUM_HEADS"] > 0 and config["MODEL_DIM"] % config["NUM_HEADS"] != 0:
        # Find nearest valid num_heads
        for nh in range(config["NUM_HEADS"], 0, -1):
            if config["MODEL_DIM"] % nh == 0:
                config["NUM_HEADS"] = nh
                break
    if config["NUM_HEADS"] % config["NUM_KV_HEADS"] != 0:
        for kvh in range(config["NUM_KV_HEADS"], 0, -1):
            if config["NUM_HEADS"] % kvh == 0:
                config["NUM_KV_HEADS"] = kvh
                break
    hd = config["MODEL_DIM"] // config["NUM_HEADS"]
    if hd % 2 != 0:
        config["NUM_HEADS"] = max(config["NUM_HEADS"] // 2, 1)
    return config


def mutate_config(config, mutation_rate=0.3):
    """Mutate a config by changing a few parameters."""
    from runner import default_config
    new = dict(config)
    base = default_config()
    for k, (lo, hi, step) in SEARCH_SPACE.items():
        if random.random() < mutation_rate:
            n_steps = int((hi - lo) / step)
            # Small perturbation from current value
            current = new.get(k, base.get(k, lo))
            current_step = int((current - lo) / step)
            delta = random.choice([-2, -1, 1, 2])
            new_step = max(0, min(n_steps, current_step + delta))
            val = lo + new_step * step
            if isinstance(base.get(k), int):
                val = int(val)
            new[k] = val
    # Re-apply constraints
    if new["NUM_KV_HEADS"] > new["NUM_HEADS"]:
        new["NUM_KV_HEADS"] = new["NUM_HEADS"]
    if new["MODEL_DIM"] % new["NUM_HEADS"] != 0:
        for nh in range(new["NUM_HEADS"], 0, -1):
            if new["MODEL_DIM"] % nh == 0:
                new["NUM_HEADS"] = nh
                break
    if new["NUM_HEADS"] % new["NUM_KV_HEADS"] != 0:
        for kvh in range(new["NUM_KV_HEADS"], 0, -1):
            if new["NUM_HEADS"] % kvh == 0:
                new["NUM_KV_HEADS"] = kvh
                break
    return new


def suggest_configs(n=5):
    """Suggest n configs that are predicted to have the best BPB."""
    results = load_results()

    if len(results) < 3:
        # Not enough data — return random configs that fit in 16MB
        print("Not enough experiment data for ML prediction. Generating diverse candidates.")
        suggestions = []
        for _ in range(n * 20):
            cfg = random_config()
            if estimate_model_size_mb(cfg) <= 15.5:
                suggestions.append((cfg, 1.3))  # placeholder score
            if len(suggestions) >= n:
                break
        return suggestions[:n]

    # Build training data
    X = np.array([config_to_features(r["config"]) for r in results])
    y = np.array([r["val_bpb"] for r in results])

    try:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=1)
    except ImportError:
        print("sklearn not available. Install with: pip install scikit-learn")
        print("Falling back to mutation-based suggestions from best result.")
        best = min(results, key=lambda r: r["val_bpb"])
        suggestions = []
        for _ in range(n):
            cfg = mutate_config(best["config"])
            if estimate_model_size_mb(cfg) <= 15.5:
                suggestions.append((cfg, best["val_bpb"]))
        return suggestions

    model.fit(X, y)

    # Generate candidates: mutations of best + random
    best_results = sorted(results, key=lambda r: r["val_bpb"])[:3]
    candidates = []
    # Mutations of top configs
    for br in best_results:
        for _ in range(200):
            cfg = mutate_config(br["config"], mutation_rate=0.2)
            if estimate_model_size_mb(cfg) <= 15.5:
                candidates.append(cfg)
    # Random exploration
    for _ in range(200):
        cfg = random_config()
        if estimate_model_size_mb(cfg) <= 15.5:
            candidates.append(cfg)

    if not candidates:
        return []

    # Predict BPB for all candidates
    X_cand = np.array([config_to_features(c) for c in candidates])
    predictions = model.predict(X_cand)

    # Pick top n by predicted BPB (lower is better)
    indices = np.argsort(predictions)[:n]
    return [(candidates[i], predictions[i]) for i in indices]


def analyze():
    """Analyze results and show feature importance."""
    results = load_results()
    if len(results) < 3:
        print(f"Only {len(results)} results. Need at least 3 for analysis.")
        return

    X = np.array([config_to_features(r["config"]) for r in results])
    y = np.array([r["val_bpb"] for r in results])

    print(f"\n{'='*60}")
    print(f"Experiment Analysis ({len(results)} runs)")
    print(f"{'='*60}")
    print(f"Best BPB:  {y.min():.4f}")
    print(f"Worst BPB: {y.max():.4f}")
    print(f"Mean BPB:  {y.mean():.4f}")
    print(f"Std BPB:   {y.std():.4f}")

    try:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        importances = sorted(
            zip(FEATURE_NAMES, model.feature_importances_),
            key=lambda x: -x[1]
        )
        print(f"\nFeature Importance (what matters most for BPB):")
        print(f"{'Feature':<30} {'Importance':<10}")
        print(f"{'-'*40}")
        for name, imp in importances:
            bar = "#" * int(imp * 50)
            print(f"{name:<30} {imp:<10.4f} {bar}")

        # Show correlation direction for top features
        print(f"\nTop feature correlations:")
        for name, _ in importances[:5]:
            idx = FEATURE_NAMES.index(name)
            corr = np.corrcoef(X[:, idx], y)[0, 1]
            direction = "higher=worse" if corr > 0 else "higher=better"
            print(f"  {name}: r={corr:.3f} ({direction})")

    except ImportError:
        print("\nInstall scikit-learn for feature importance analysis: pip install scikit-learn")


def main():
    parser = argparse.ArgumentParser(description="Parameter Golf Predictor")
    parser.add_argument("--suggest", type=int, default=0, help="Suggest N configs")
    parser.add_argument("--analyze", action="store_true", help="Analyze results")
    args = parser.parse_args()

    if args.analyze:
        analyze()
    elif args.suggest > 0:
        suggestions = suggest_configs(n=args.suggest)
        for i, (cfg, score) in enumerate(suggestions, 1):
            from runner import default_config
            diff = {k: v for k, v in cfg.items() if v != default_config().get(k)}
            print(f"\nSuggestion {i} (predicted BPB: {score:.4f}):")
            print(f"  Changes: {json.dumps(diff, indent=2)}")
            print(f"  Est. size: {estimate_model_size_mb(cfg):.2f}MB")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
