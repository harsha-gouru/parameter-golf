"""
Modal runner for Parameter Golf training.
Usage:
    # 1×H100 quick test:
    modal run modal_train.py --run-id test1 --gpus 1

    # 8×H100 full run:
    modal run modal_train.py --run-id sota_v1 --gpus 8

    # Custom config:
    modal run modal_train.py --run-id depth5 --gpus 1 --depth-recurrence 5 --num-layers 3 --model-dim 896

    # Dry run:
    modal run modal_train.py --dry-run
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("parameter-golf")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "sentencepiece", "huggingface-hub")
)

vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

DEFAULTS = {
    "NUM_LAYERS": 9, "MODEL_DIM": 512, "NUM_HEADS": 8, "NUM_KV_HEADS": 4,
    "MLP_MULT": 3, "VOCAB_SIZE": 1024, "TIE_EMBEDDINGS": 1,
    "ROPE_BASE": 10000.0, "LOGIT_SOFTCAP": 30.0, "QK_GAIN_INIT": 1.5,
    "TIED_EMBED_INIT_STD": 0.005, "TRAIN_SEQ_LEN": 1024,
    "TRAIN_BATCH_TOKENS": 524288, "ITERATIONS": 20000, "WARMDOWN_ITERS": 3000,
    "WARMUP_STEPS": 20, "MAX_WALLCLOCK_SECONDS": 600.0,
    "TIED_EMBED_LR": 0.03, "MATRIX_LR": 0.02, "SCALAR_LR": 0.02,
    "MUON_MOMENTUM": 0.99, "MUON_BACKEND_STEPS": 5,
    "MUON_MOMENTUM_WARMUP_START": 0.92, "MUON_MOMENTUM_WARMUP_STEPS": 1500,
    "BETA1": 0.9, "BETA2": 0.95, "ADAM_EPS": 1e-08, "GRAD_CLIP_NORM": 0.3,
    "WEIGHT_DECAY": 0.01, "MUON_WEIGHT_DECAY": 0.02,
    "QAT_ENABLED": 1, "QAT_START_FRAC": 0.0, "QAT_RAMP_FRAC": 0.1,
    "COSINE_WARMDOWN": 1, "DEPTH_RECURRENCE": 1,
    "FP16_EMBED": 1, "EVAL_STRIDE": 64, "QUANT_BITS": 6, "ORTHO_INIT": 1,
    "SEED": 1337, "VAL_LOSS_EVERY": 0, "TRAIN_LOG_EVERY": 100,
}


def parse_results(output: str) -> dict:
    results = {}
    for line in output.splitlines():
        if "final_int8_zlib_roundtrip_exact" in line:
            for part in line.split():
                if part.startswith("val_bpb:"): results["val_bpb"] = float(part.split(":")[1])
                if part.startswith("val_loss:"): results["val_loss"] = float(part.split(":")[1])
        if "final_sliding_window_exact" in line:
            for part in line.split():
                if part.startswith("val_bpb:"): results["sliding_window_bpb"] = float(part.split(":")[1])
        if "Total submission size int8+zlib:" in line:
            results["model_size_bytes"] = int(line.split(":")[1].strip().split()[0])
        if "val_bpb:" in line and "final_" not in line and "roundtrip" not in line and "sliding" not in line and "enabled" not in line:
            for part in line.split():
                if part.startswith("val_bpb:"): results["pre_quant_bpb"] = float(part.split(":")[1])
        if "step_avg:" in line:
            for part in line.split():
                if part.startswith("step_avg:"): results["ms_per_step"] = float(part.rstrip("ms").split(":")[1])
        if "stopping_early" in line:
            for part in line.split():
                if part.startswith("step:"): results["final_step"] = part.split(":")[1]
    return results


# We create separate functions for different GPU counts since Modal
# requires gpu= to be set at decoration time.

@app.function(image=image, gpu="H100:1", timeout=1800, volumes={"/data": vol})
def train_1gpu(run_id: str, config: dict) -> dict:
    return _train(run_id, config, nproc=1)

@app.function(image=image, gpu="H100:2", timeout=1800, volumes={"/data": vol})
def train_2gpu(run_id: str, config: dict) -> dict:
    return _train(run_id, config, nproc=2)

@app.function(image=image, gpu="H100:4", timeout=1800, volumes={"/data": vol})
def train_4gpu(run_id: str, config: dict) -> dict:
    return _train(run_id, config, nproc=4)

@app.function(image=image, gpu="H100:8", timeout=1800, volumes={"/data": vol})
def train_8gpu(run_id: str, config: dict) -> dict:
    return _train(run_id, config, nproc=8)


def _train(run_id: str, config: dict, nproc: int) -> dict:
    """Core training logic — streams output live."""
    import shutil

    work_dir = Path("/workspace/parameter-golf")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy train script
    script_src = Path("/data/train_gpt.py")
    script_dst = work_dir / "train_gpt.py"
    if script_src.exists():
        shutil.copy2(script_src, script_dst)
    else:
        raise FileNotFoundError("train_gpt.py not in /data. Run modal_upload.py first.")

    # Dataset
    data_dir = Path("/data/data/datasets/fineweb10B_sp1024")
    tok_dir = Path("/data/data/tokenizers")
    if not data_dir.exists() or not list(data_dir.glob("fineweb_val_*.bin")):
        raise FileNotFoundError("Dataset not found. Run modal_upload.py first.")

    # Environment
    env = {k: str(v) for k, v in config.items()}
    env["RUN_ID"] = run_id
    env["DATA_PATH"] = str(data_dir)
    env["TOKENIZER_PATH"] = str(tok_dir / "fineweb_1024_bpe.model")
    full_env = {**os.environ, **env}

    print(f"=== {run_id} | {nproc}×H100 | {config.get('NUM_LAYERS')}L dim={config.get('MODEL_DIM')} MLP{config.get('MLP_MULT')}x DR={config.get('DEPTH_RECURRENCE')} Q{config.get('QUANT_BITS')} ===")

    # Stream output live via Popen instead of subprocess.run
    t0 = time.time()
    output_lines = []
    proc = subprocess.Popen(
        ["torchrun", "--standalone", f"--nproc_per_node={nproc}", str(script_dst)],
        env=full_env, cwd=str(work_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    for line in proc.stdout:
        line = line.rstrip()
        output_lines.append(line)
        # Print key lines live (skip noisy warmup)
        if any(k in line for k in ["step:", "val_bpb:", "val_loss:", "final_", "stopping", "peak memory", "Serialized", "Total submission", "Error", "Traceback"]):
            print(line, flush=True)

    proc.wait()
    elapsed = time.time() - t0
    output = "\n".join(output_lines)

    # Parse results
    metrics = parse_results(output)

    # Save full log
    log_path = Path(f"/data/logs/{run_id}.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output)

    record = {
        "run_id": run_id,
        "config": config,
        "val_bpb": metrics.get("val_bpb"),
        "pre_quant_bpb": metrics.get("pre_quant_bpb"),
        "sliding_window_bpb": metrics.get("sliding_window_bpb"),
        "val_loss": metrics.get("val_loss"),
        "model_size_bytes": metrics.get("model_size_bytes"),
        "elapsed_seconds": elapsed,
        "ms_per_step": metrics.get("ms_per_step"),
        "final_step": metrics.get("final_step"),
        "exit_code": proc.returncode,
        "status": "success" if proc.returncode == 0 and metrics.get("val_bpb") else "failed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nproc": nproc,
    }

    # Save to volume
    with open(Path("/data/results.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")

    vol.commit()

    print(f"\n=== DONE: {run_id} ===")
    print(f"Pre-quant BPB:  {record.get('pre_quant_bpb', 'N/A')}")
    print(f"Post-quant BPB: {record.get('val_bpb', 'N/A')}")
    print(f"Sliding BPB:    {record.get('sliding_window_bpb', 'N/A')}")
    print(f"Artifact:       {record.get('model_size_bytes', 'N/A')} bytes")
    print(f"Elapsed:        {elapsed:.1f}s | Exit: {proc.returncode}")

    return record


@app.local_entrypoint()
def main(
    run_id: str = "sota_baseline",
    gpus: int = 1,
    dry_run: bool = False,
    num_layers: int = DEFAULTS["NUM_LAYERS"],
    model_dim: int = DEFAULTS["MODEL_DIM"],
    mlp_mult: int = DEFAULTS["MLP_MULT"],
    depth_recurrence: int = DEFAULTS["DEPTH_RECURRENCE"],
    quant_bits: int = DEFAULTS["QUANT_BITS"],
    fp16_embed: int = DEFAULTS["FP16_EMBED"],
    matrix_lr: float = DEFAULTS["MATRIX_LR"],
    warmdown_iters: int = DEFAULTS["WARMDOWN_ITERS"],
    train_seq_len: int = DEFAULTS["TRAIN_SEQ_LEN"],
    seed: int = DEFAULTS["SEED"],
    eval_stride: int = DEFAULTS["EVAL_STRIDE"],
    qat_enabled: int = DEFAULTS["QAT_ENABLED"],
):
    config = dict(DEFAULTS)
    config.update({
        "NUM_LAYERS": num_layers, "MODEL_DIM": model_dim, "MLP_MULT": mlp_mult,
        "DEPTH_RECURRENCE": depth_recurrence, "QUANT_BITS": quant_bits,
        "FP16_EMBED": fp16_embed, "MATRIX_LR": matrix_lr,
        "WARMDOWN_ITERS": warmdown_iters, "TRAIN_SEQ_LEN": train_seq_len,
        "SEED": seed, "EVAL_STRIDE": eval_stride, "QAT_ENABLED": qat_enabled,
    })

    cost_per_hr = {1: 3.95, 2: 7.90, 4: 15.80, 8: 31.60}.get(gpus, gpus * 3.95)
    cost_10min = cost_per_hr / 6

    if dry_run:
        print(f"DRY RUN: {run_id} | {gpus}×H100 (~${cost_10min:.2f}/run)")
        print(json.dumps(config, indent=2))
        return

    print(f"Launching: {run_id} | {gpus}×H100 (~${cost_10min:.2f}/run)")

    train_fn = {1: train_1gpu, 2: train_2gpu, 4: train_4gpu, 8: train_8gpu}[gpus]
    result = train_fn.remote(run_id, config)

    # Save locally
    with open(Path("experiments/results.jsonl"), "a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"\nSaved to experiments/results.jsonl")
