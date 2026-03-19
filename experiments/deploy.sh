#!/bin/bash
# Deploy parameter-golf to a remote GPU instance.
# Usage: bash experiments/deploy.sh user@host
#
# This syncs the repo, sets up deps, and starts the sweep.
# The instance stays alive between runs — just re-run to sync changes.

set -e

if [ -z "$1" ]; then
    echo "Usage: bash experiments/deploy.sh user@host [--run]"
    echo ""
    echo "Options:"
    echo "  --run       Start the sweep after deploying"
    echo "  --short     Use short runs (2 min each)"
    echo "  --nproc N   GPUs per run (default: 1)"
    echo ""
    echo "Examples:"
    echo "  bash experiments/deploy.sh root@203.0.113.5"
    echo "  bash experiments/deploy.sh root@203.0.113.5 --run"
    echo "  bash experiments/deploy.sh root@203.0.113.5 --run --short"
    exit 1
fi

HOST="$1"
shift

RUN=false
SHORT=""
NPROC=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run) RUN=true; shift ;;
        --short) SHORT="--short"; shift ;;
        --nproc) NPROC="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="/workspace/parameter-golf"

echo "=== Syncing to $HOST:$REMOTE_DIR ==="
rsync -avz --progress \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'data/datasets' \
    --exclude 'data/tokenizers' \
    --exclude 'logs' \
    --exclude '.git' \
    "$REPO_DIR/" "$HOST:$REMOTE_DIR/"

echo ""
echo "=== Setting up on remote ==="
ssh "$HOST" "cd $REMOTE_DIR && bash experiments/gpu_setup.sh"

if [ "$RUN" = true ]; then
    echo ""
    echo "=== Starting sweep (nproc=$NPROC $SHORT) ==="
    ssh "$HOST" "cd $REMOTE_DIR && nohup python experiments/run_sweep.py --nproc $NPROC $SHORT > sweep.log 2>&1 &"
    echo "Sweep started in background. Monitor with:"
    echo "  ssh $HOST 'tail -f $REMOTE_DIR/sweep.log'"
    echo ""
    echo "Pull results back with:"
    echo "  scp $HOST:$REMOTE_DIR/experiments/results.jsonl experiments/results.jsonl"
fi

echo ""
echo "=== Manual commands ==="
echo "  ssh $HOST"
echo "  cd $REMOTE_DIR"
echo "  python experiments/run_sweep.py --nproc $NPROC $SHORT"
echo ""
echo "Pull results:"
echo "  scp $HOST:$REMOTE_DIR/experiments/results.jsonl experiments/results.jsonl"
echo "  python experiments/runner.py --leaderboard"
