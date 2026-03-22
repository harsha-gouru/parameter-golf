"""Upload train_gpt.py and data scripts to Modal volume, then download dataset."""
import modal
from pathlib import Path

app = modal.App("parameter-golf-upload")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "sentencepiece", "huggingface-hub")
)

vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def upload_and_download_data(train_script: bytes, cache_script: bytes):
    """Upload scripts and download the FineWeb dataset."""
    # Save train_gpt.py at volume root
    Path("/data/train_gpt.py").write_bytes(train_script)
    print(f"Uploaded train_gpt.py ({len(train_script):,} bytes)")

    # The download script uses ROOT = Path(__file__).parent for output.
    # Place it at /data/data/ so it creates /data/data/datasets/ and /data/data/tokenizers/
    data_subdir = Path("/data/data")
    data_subdir.mkdir(exist_ok=True)
    cache_path = data_subdir / "cached_challenge_fineweb.py"
    cache_path.write_bytes(cache_script)
    print(f"Uploaded cached_challenge_fineweb.py to {cache_path}")

    # Check if dataset already exists
    ds_dir = data_subdir / "datasets" / "fineweb10B_sp1024"
    val_files = list(ds_dir.glob("fineweb_val_*.bin")) if ds_dir.exists() else []
    train_files = list(ds_dir.glob("fineweb_train_*.bin")) if ds_dir.exists() else []

    if val_files and len(train_files) >= 80:
        print(f"Dataset already cached: {len(train_files)} train shards, {len(val_files)} val shards")
        vol.commit()
        return

    print("Downloading FineWeb dataset (80 train shards)... This takes ~5 minutes.")
    import subprocess
    result = subprocess.run(
        ["python3", str(cache_path), "--variant", "sp1024", "--train-shards", "80"],
        capture_output=True, text=True, timeout=1500,
    )
    if result.stdout:
        print(result.stdout[-2000:])
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-2000:]}")
        raise RuntimeError("Dataset download failed")

    # Verify
    train_files = list(ds_dir.glob("fineweb_train_*.bin"))
    val_files = list(ds_dir.glob("fineweb_val_*.bin"))
    tok_files = list((data_subdir / "tokenizers").glob("*"))
    print(f"Dataset ready: {len(train_files)} train, {len(val_files)} val, {len(tok_files)} tokenizer files")

    total_size = sum(f.stat().st_size for f in data_subdir.rglob("*") if f.is_file())
    print(f"Total data size: {total_size / 1e9:.2f} GB")

    vol.commit()


@app.local_entrypoint()
def main():
    print("Uploading files and downloading dataset to Modal volume...")
    train_script = Path("train_gpt.py").read_bytes()
    cache_script = Path("data/cached_challenge_fineweb.py").read_bytes()
    upload_and_download_data.remote(train_script, cache_script)
    print("Done! Volume is ready for training runs.")
