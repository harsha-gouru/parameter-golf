#!/bin/bash
# Run this on your PrimeIntellect GPU instance.
# Usage: cd /workspace/parameter-golf && bash experiments/gpu_setup.sh
set -e

echo "=== Parameter Golf GPU Setup ==="

pip install -q scikit-learn 2>/dev/null || true

# Download sp1024 dataset (full 80 shards)
if [ ! -f "data/datasets/fineweb10B_sp1024/fineweb_train_000079.bin" ]; then
    echo "Downloading FineWeb sp1024 (full)..."
    python3 data/cached_challenge_fineweb.py --variant sp1024
fi

# Download sp2048 dataset for bigger vocab experiments
if [ ! -f "data/datasets/fineweb10B_sp2048/fineweb_val_000000.bin" ]; then
    echo "Downloading FineWeb sp2048 (full)..."
    python3 data/cached_challenge_fineweb.py --variant sp2048
fi

echo "=== Setup complete ==="
echo ""
echo "Run sweep:    python experiments/run_sweep.py"
echo "Leaderboard:  python experiments/runner.py --leaderboard"
echo "Suggest next: python experiments/runner.py --suggest --n 5"
