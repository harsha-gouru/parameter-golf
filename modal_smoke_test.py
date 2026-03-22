"""
Modal training runner for parameter-golf submission.
Dataset is cached in a persistent volume — only downloaded once.

Usage:
  modal run modal_smoke_test.py                        # cache data + 200-step smoke test
  modal run modal_smoke_test.py --cache-only           # just cache dataset
  modal run modal_smoke_test.py --max-steps 500        # longer test
  modal run modal_smoke_test.py --full                 # full 10-min 8×H100 run
  modal run modal_smoke_test.py --wandb-key sk-...     # enable wandb logging
"""
import os
import modal

# Load .env file if present (for WANDB_API_KEY etc)
_env_file = os.path.expanduser("~/.env")
if os.path.exists(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

_WANDB_KEY = os.environ.get("WANDB_API_KEY", "")

app = modal.App("parameter-golf")

# Persistent volume — dataset cached here, reused across ALL runs
dataset_vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

# Image: bake the submission script + deps into the container
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5",
        "numpy",
        "sentencepiece",
        "huggingface_hub",
        "zstandard",
        "wandb",
    )
    .add_local_file("train_gpt_submission.py", "/code/train_gpt.py")
)

DATASET_PATH = "/data"


# ------------------------------------------------------------------
# 1. DATASET CACHING (runs once, persists forever)
# ------------------------------------------------------------------

@app.function(
    image=image,
    volumes={DATASET_PATH: dataset_vol},
    timeout=600,
    cpu=4,
    memory=16384,
)
def cache_dataset(variant: str = "sp1024", train_shards: int = 80):
    """Download and cache dataset in Modal volume. Only runs if not already cached."""
    import os
    import glob as globmod

    ds_dir = f"{DATASET_PATH}/datasets/fineweb10B_{variant}"
    tok_dir = f"{DATASET_PATH}/tokenizers"

    existing_train = sorted(globmod.glob(f"{ds_dir}/fineweb_train_*.bin"))
    existing_val = sorted(globmod.glob(f"{ds_dir}/fineweb_val_*.bin"))
    existing_tok = globmod.glob(f"{tok_dir}/*{variant.replace('sp', '')}*.model")

    if len(existing_train) >= train_shards and existing_val and existing_tok:
        print(f"Already cached: {len(existing_train)} train, {len(existing_val)} val shards, tok={existing_tok[0]}")
        return {"cached": True, "train": len(existing_train), "val": len(existing_val)}

    print(f"Downloading: variant={variant}, train_shards={train_shards}...")
    from huggingface_hub import hf_hub_download, list_repo_files

    REPO_ID = "willdepueoai/parameter-golf"
    os.makedirs(ds_dir, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)

    files = list_repo_files(REPO_ID, repo_type="dataset")

    # Tokenizer
    for f in [f for f in files if "tokenizers/" in f and variant.replace("sp", "") in f]:
        print(f"  tok: {f}")
        hf_hub_download(REPO_ID, f, repo_type="dataset", local_dir=DATASET_PATH)

    # Validation (all)
    val_files = sorted([f for f in files if f"fineweb10B_{variant}/fineweb_val_" in f])
    for f in val_files:
        local = f"{DATASET_PATH}/{f}"
        if not os.path.exists(local):
            print(f"  val: {f}")
            hf_hub_download(REPO_ID, f, repo_type="dataset", local_dir=DATASET_PATH)

    # Training (limited)
    train_files = sorted([f for f in files if f"fineweb10B_{variant}/fineweb_train_" in f])
    for f in train_files[:train_shards]:
        local = f"{DATASET_PATH}/{f}"
        if not os.path.exists(local):
            print(f"  train: {f}")
            hf_hub_download(REPO_ID, f, repo_type="dataset", local_dir=DATASET_PATH)

    dataset_vol.commit()
    result = {"cached": True, "train": min(train_shards, len(train_files)), "val": len(val_files)}
    print(f"Done: {result}")
    return result


# ------------------------------------------------------------------
# 2. TRAINING (single GPU smoke test or multi-GPU full run)
# ------------------------------------------------------------------

@app.function(
    image=image,
    gpu="H100",
    volumes={DATASET_PATH: dataset_vol},
    timeout=1200,
    memory=65536,
)
def train(
    max_steps: int = 200,
    max_wallclock: float = 120.0,
    run_id: str = "smoke",
    wandb_key: str = "",
    # Architecture
    num_layers: int = 10,
    model_dim: int = 512,
    mlp_mult: str = "3",
    # Our features
    bigram_logit: bool = False,
    # Eval
    eval_stride: int = 0,
):
    """Run training on a single H100. Data must already be cached."""
    import os
    import sys
    import subprocess
    import glob as globmod
    import json
    import time

    # ---- Reload volume to pick up any recent writes ----
    dataset_vol.reload()

    # ---- Debug: find actual file locations ----
    import subprocess as _sp
    find_result = _sp.run(["find", DATASET_PATH, "-name", "*.bin", "-o", "-name", "*.model"],
                          capture_output=True, text=True, timeout=30)
    print(f"Files in volume:\n{find_result.stdout[:3000]}")

    # ---- Verify dataset ----
    train_shards = sorted(globmod.glob(f"{DATASET_PATH}/**/fineweb_train_*.bin", recursive=True))
    val_shards = sorted(globmod.glob(f"{DATASET_PATH}/**/fineweb_val_*.bin", recursive=True))
    tok_files = globmod.glob(f"{DATASET_PATH}/**/*1024*.model", recursive=True)

    if not train_shards or not val_shards or not tok_files:
        raise RuntimeError(
            f"Dataset not cached! train={len(train_shards)}, val={len(val_shards)}, tok={len(tok_files)}. "
            "Run: modal run modal_smoke_test.py --cache-only"
        )

    tok_path = tok_files[0]
    # Infer the dataset directory from the first train shard
    import os.path
    data_dir = os.path.dirname(train_shards[0])
    print(f"Dataset: {len(train_shards)} train, {len(val_shards)} val")
    print(f"  data_dir: {data_dir}")
    print(f"  tokenizer: {tok_path}")

    # ---- Wandb setup ----
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
        os.environ["WANDB_PROJECT"] = "parameter-golf"
        os.environ["WANDB_RUN_NAME"] = run_id
        print(f"Wandb: enabled (project=parameter-golf, run={run_id})")

    # ---- Build env ----
    env = {k: v for k, v in os.environ.items()}
    env.update({
        "DATA_PATH": data_dir,
        "TOKENIZER_PATH": tok_path,
        "VOCAB_SIZE": "1024",
        "RUN_ID": run_id,
        # Architecture
        "NUM_LAYERS": str(num_layers),
        "MODEL_DIM": str(model_dim),
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": mlp_mult,
        "TIE_EMBEDDINGS": "1",
        # Training
        "ITERATIONS": str(max_steps),
        "MAX_WALLCLOCK_SECONDS": str(max_wallclock),
        "WARMUP_STEPS": "5",
        "WARMDOWN_ITERS": str(max(max_steps // 6, 50)),
        "TRAIN_BATCH_TOKENS": "65536",
        "TRAIN_SEQ_LEN": "1024",
        "TRAIN_LOG_EVERY": "10",
        "VAL_LOSS_EVERY": str(max(max_steps // 4, 50)),
        # Optimizer
        "MATRIX_LR": "0.025",
        "SCALAR_LR": "0.025",
        "TIED_EMBED_LR": "0.035",
        "WEIGHT_DECAY": "0.04",
        "MUON_MOMENTUM": "0.99",
        "MUON_MOMENTUM_WARMUP_START": "0.92",
        "MUON_MOMENTUM_WARMUP_STEPS": "300",
        "GRAD_CLIP_NORM": "0.3",
        # N-gram embeddings
        "BIGRAM_VOCAB_SIZE": "4096",
        "BIGRAM_DIM": "128",
        "BIGRAM_LOGIT_HEAD": "1" if bigram_logit else "0",
        # Architectural improvements
        "XSA_LAST_N": "4",
        "ROPE_DIMS": "16",
        "LN_SCALE": "1",
        # Eval (disable EMA for short smoke tests — needs 1000+ steps)
        "EMA_ENABLED": "1" if max_steps >= 1000 else "0",
        "EVAL_STRIDE": str(eval_stride),
    })

    # Remove DDP env vars — single GPU mode
    for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
                "TORCHELASTIC_RUN_ID", "GROUP_RANK", "GROUP_WORLD_SIZE"]:
        env.pop(key, None)

    # ---- Print config summary ----
    print(f"\n{'='*80}")
    print(f"RUN: {run_id}")
    print(f"  arch: {num_layers}L, dim={model_dim}, mlp={mlp_mult}, heads=8/4")
    print(f"  training: {max_steps} steps, {max_wallclock}s wallclock, batch=65K, seq=1024")
    print(f"  features: bigram_logit={bigram_logit}, BigramHash(2048), XSA4, PartialRoPE16, LN_Scale, EMA")
    print(f"  eval: sliding_stride={eval_stride}")
    print(f"{'='*80}\n")

    t_start = time.time()

    # ---- Run training ----
    result = subprocess.run(
        [sys.executable, "/code/train_gpt.py"],
        env=env,
        capture_output=True,
        text=True,
        timeout=int(max_wallclock + 300),
        cwd="/tmp",
    )

    elapsed = time.time() - t_start

    # ---- Print output ----
    print(result.stdout[-10000:] if len(result.stdout) > 10000 else result.stdout)
    if result.returncode != 0:
        print(f"\n{'='*80}\nSTDERR (last 5000 chars):\n{'='*80}")
        print(result.stderr[-5000:])
        raise RuntimeError(f"Training failed (exit code {result.returncode})")

    # ---- Check artifact ----
    artifact = "/tmp/final_model.int8.ptz"
    summary = {"run_id": run_id, "exit_code": result.returncode, "elapsed_s": round(elapsed, 1)}

    if os.path.exists(artifact):
        size = os.path.getsize(artifact)
        code_size = len(open("/code/train_gpt.py").read().encode("utf-8"))
        total = size + code_size
        limit = 16_000_000

        summary.update({
            "artifact_bytes": size,
            "code_bytes": code_size,
            "total_bytes": total,
            "under_limit": total <= limit,
            "headroom_bytes": limit - total,
        })

        print(f"\n{'='*80}")
        print(f"ARTIFACT: {size:,} bytes ({size/1e6:.2f} MB)")
        print(f"CODE:     {code_size:,} bytes")
        print(f"TOTAL:    {total:,} bytes ({total/1e6:.2f} MB)")
        print(f"LIMIT:    {limit:,} bytes")
        print(f"STATUS:   {'PASS' if total <= limit else 'FAIL - OVER LIMIT'}")
        print(f"HEADROOM: {(limit-total)/1e6:.2f} MB")
        print(f"{'='*80}")
    else:
        print("WARNING: No artifact file found!")

    # ---- Extract final metrics from stdout ----
    for line in result.stdout.split("\n"):
        if "final_int8_zlib_roundtrip_exact" in line:
            summary["final_line"] = line.strip()
            # Parse val_bpb
            for part in line.split():
                if part.startswith("val_bpb:"):
                    summary["val_bpb"] = float(part.split(":")[1])

    print(f"\nSummary: {json.dumps(summary, indent=2)}")
    return summary


# ------------------------------------------------------------------
# 3. FULL 8xH100 TRAINING (competition submission)
# ------------------------------------------------------------------

@app.function(
    image=image,
    gpu="H100:8",
    volumes={DATASET_PATH: dataset_vol},
    timeout=1800,
    memory=262144,
)
def train_8gpu(
    run_id: str = "full-8gpu",
    wandb_key: str = "",
    num_layers: int = 10,
    model_dim: int = 512,
    mlp_mult: str = "3",
    bigram_logit: bool = False,
):
    """Full 10-min training on 8×H100 with torchrun DDP."""
    import os
    import sys
    import subprocess
    import glob as globmod
    import json
    import time

    dataset_vol.reload()

    train_shards = sorted(globmod.glob(f"{DATASET_PATH}/**/fineweb_train_*.bin", recursive=True))
    val_shards = sorted(globmod.glob(f"{DATASET_PATH}/**/fineweb_val_*.bin", recursive=True))
    tok_files = globmod.glob(f"{DATASET_PATH}/**/*1024*.model", recursive=True)

    if not train_shards or not val_shards or not tok_files:
        raise RuntimeError(f"Dataset not cached! train={len(train_shards)}, val={len(val_shards)}, tok={len(tok_files)}")

    tok_path = tok_files[0]
    data_dir = os.path.dirname(train_shards[0])
    print(f"Dataset: {len(train_shards)} train, {len(val_shards)} val | data_dir: {data_dir}")

    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
        os.environ["WANDB_PROJECT"] = "parameter-golf"
        os.environ["WANDB_RUN_NAME"] = run_id

    env = {k: v for k, v in os.environ.items()}
    env.update({
        "DATA_PATH": data_dir,
        "TOKENIZER_PATH": tok_path,
        "VOCAB_SIZE": "1024",
        "RUN_ID": run_id,
        # Architecture
        "NUM_LAYERS": str(num_layers),
        "MODEL_DIM": str(model_dim),
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": mlp_mult,
        "TIE_EMBEDDINGS": "1",
        # Training — full competition settings
        "ITERATIONS": "20000",
        "MAX_WALLCLOCK_SECONDS": "600",
        "WARMUP_STEPS": "20",
        "WARMDOWN_ITERS": "2800",
        "TRAIN_BATCH_TOKENS": "786432",
        "TRAIN_SEQ_LEN": "2048",
        "TRAIN_LOG_EVERY": "100",
        "VAL_LOSS_EVERY": "500",
        # Optimizer
        "MATRIX_LR": "0.025",
        "SCALAR_LR": "0.025",
        "TIED_EMBED_LR": "0.035",
        "WEIGHT_DECAY": "0.04",
        "MUON_MOMENTUM": "0.99",
        "MUON_MOMENTUM_WARMUP_START": "0.92",
        "MUON_MOMENTUM_WARMUP_STEPS": "1500",
        "GRAD_CLIP_NORM": "0.3",
        # Features
        "BIGRAM_LOGIT_HEAD": "1" if bigram_logit else "0",
        "BIGRAM_VOCAB_SIZE": "4096",
        "BIGRAM_DIM": "128",
        # Architectural improvements
        "XSA_LAST_N": "4",
        "ROPE_DIMS": "16",
        "LN_SCALE": "1",
        # Eval
        "SWA_ENABLED": "1",
        "SWA_START_FRAC": "0.4",
        "SWA_EVERY": "50",
        "EVAL_STRIDE": "64",
        "EVAL_BATCH_SEQS": "32",
    })

    print(f"\n{'='*80}")
    print(f"FULL 8×H100 RUN: {run_id}")
    print(f"  arch: {num_layers}L, dim={model_dim}, mlp={mlp_mult}")
    print(f"  training: 20K iters, 600s wallclock, batch=786K, seq=2048")
    print(f"  features: bigram_logit={bigram_logit}, BigramHash(2048), XSA4, PartialRoPE16, LN_Scale, EMA")
    print(f"{'='*80}\n")

    # Verify GPUs before launching torchrun
    import torch as _torch
    gpu_count = _torch.cuda.device_count()
    print(f"CUDA devices visible: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i}: {_torch.cuda.get_device_name(i)}")
    if gpu_count < 8:
        raise RuntimeError(f"Expected 8 GPUs, got {gpu_count}")

    env["NCCL_IB_DISABLE"] = "1"  # disable InfiniBand (single node, use NVLink)

    t_start = time.time()

    # Stream output live instead of capturing (so we see progress)
    logfile = "/tmp/train_8gpu.log"
    with open(logfile, "w") as log_f:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "torch.distributed.run",
                "--standalone", "--nproc_per_node=8",
                "/code/train_gpt.py",
            ],
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd="/tmp",
        )
        # Print progress periodically while waiting
        import select
        while proc.poll() is None:
            time.sleep(30)
            elapsed_so_far = time.time() - t_start
            # Print last few lines of log
            try:
                tail = subprocess.run(["tail", "-5", logfile], capture_output=True, text=True, timeout=5)
                latest = tail.stdout.strip().split("\n")[-1] if tail.stdout.strip() else "..."
                print(f"  [{elapsed_so_far:.0f}s] {latest}", flush=True)
            except Exception:
                pass

    elapsed = time.time() - t_start
    returncode = proc.returncode

    # Read full log
    with open(logfile) as f:
        full_output = f.read()
    print(full_output[-15000:] if len(full_output) > 15000 else full_output)

    if returncode != 0:
        print(f"\n{'='*80}\nTRAINING FAILED (exit code {returncode})\n{'='*80}")
        raise RuntimeError(f"Training failed (exit code {returncode})")

    summary = {"run_id": run_id, "gpus": 8, "exit_code": returncode, "elapsed_s": round(elapsed, 1)}

    artifact = "/tmp/final_model.int8.ptz"
    if os.path.exists(artifact):
        size = os.path.getsize(artifact)
        code_size = len(open("/code/train_gpt.py").read().encode("utf-8"))
        total = size + code_size
        limit = 16_000_000
        summary.update({
            "artifact_bytes": size, "code_bytes": code_size,
            "total_bytes": total, "under_limit": total <= limit,
            "headroom_bytes": limit - total,
        })
        print(f"\n{'='*80}")
        print(f"ARTIFACT: {size:,} bytes ({size/1e6:.2f} MB)")
        print(f"TOTAL:    {total:,} bytes ({total/1e6:.2f} MB)")
        print(f"STATUS:   {'PASS' if total <= limit else 'FAIL - OVER LIMIT'}")
        print(f"HEADROOM: {(limit-total)/1e6:.2f} MB")
        print(f"{'='*80}")

    for line in full_output.split("\n"):
        if "final_int8_zlib_roundtrip_exact" in line:
            summary["final_line"] = line.strip()
            for part in line.split():
                if part.startswith("val_bpb:"):
                    summary["val_bpb"] = float(part.split(":")[1])

    print(f"\nSummary: {json.dumps(summary, indent=2)}")
    return summary


# ------------------------------------------------------------------
# 4. LOCAL ENTRYPOINT
# ------------------------------------------------------------------

@app.local_entrypoint()
def main(
    cache_only: bool = False,
    full: bool = False,
    full_8gpu: bool = False,
    train_shards: int = 80,
    max_steps: int = 200,
    max_wallclock: float = 120.0,
    run_id: str = "smoke",
    wandb_key: str = _WANDB_KEY,
    # Architecture overrides
    num_layers: int = 10,
    model_dim: int = 512,
    mlp_mult: str = "3",
    no_bigram_logit: bool = False,
):
    """
    Examples:
      modal run modal_smoke_test.py                              # smoke test (200 steps, ~2min)
      modal run modal_smoke_test.py --cache-only                 # just cache data
      modal run modal_smoke_test.py --max-steps 1000 --run-id exp1  # medium run
      modal run modal_smoke_test.py --full-8gpu --run-id sub1    # FULL 8xH100 10-min run
      modal run modal_smoke_test.py --num-layers 11 --run-id 11L   # arch sweep
    """
    # Always ensure data is cached first
    print("Checking dataset cache...")
    result = cache_dataset.remote(variant="sp1024", train_shards=train_shards)
    print(f"  -> {result}")

    if cache_only:
        print("Cache-only mode. Done.")
        return

    # Full 8×H100 competition run
    if full_8gpu:
        run_id = "full-8gpu" if run_id == "smoke" else run_id
        print(f"\n8×H100 FULL RUN: {run_id}")
        summary = train_8gpu.remote(
            run_id=run_id,
            wandb_key=wandb_key,
            num_layers=num_layers,
            model_dim=model_dim,
            mlp_mult=mlp_mult,
            bigram_logit=not no_bigram_logit,
        )
        print(f"\nFinal summary: {summary}")
        if "val_bpb" in summary:
            print(f"\n  val_bpb = {summary['val_bpb']:.6f}")
        return

    if full:
        max_steps = 20000
        max_wallclock = 600.0
        run_id = "full_10min" if run_id == "smoke" else run_id
        print(f"\nFULL RUN MODE: {max_steps} steps, {max_wallclock}s")

    print(f"\nLaunching training: run_id={run_id}, steps={max_steps}, wallclock={max_wallclock}s")

    summary = train.remote(
        max_steps=max_steps,
        max_wallclock=max_wallclock,
        run_id=run_id,
        wandb_key=wandb_key,
        num_layers=num_layers,
        model_dim=model_dim,
        mlp_mult=mlp_mult,
        bigram_logit=not no_bigram_logit,
        eval_stride=64 if full else 0,
    )

    print(f"\nFinal summary: {summary}")
    if "val_bpb" in summary:
        print(f"\n  val_bpb = {summary['val_bpb']:.6f}")
