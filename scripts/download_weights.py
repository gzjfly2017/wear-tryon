#!/usr/bin/env python3
"""Download Mobile-VTON checkpoint files from hf-mirror (HuggingFace mirror).

Uses curl (available on Windows/macOS/Linux) because Python urllib is
significantly slower against hf-mirror.

Usage:
    python scripts/download_weights.py [--out Models/mobile-vton] [--mirror https://hf-mirror.com]
"""
import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

MIRROR = "https://hf-mirror.com"
REPO = "FlashStight/Mobile-VTON"


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "download-weights/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} (exists)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get ] {url}")
    r = subprocess.run(
        ["curl", "-sL", "--retry", "3", "--connect-timeout", "30", "-o", str(tmp), url],
        check=False,
    )
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        print(f"  [fail] {dest.name}", file=sys.stderr)
        sys.exit(1)
    tmp.rename(dest)
    print(f"  [done] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Models/mobile-vton")
    ap.add_argument("--mirror", default=MIRROR)
    args = ap.parse_args()

    out = Path(args.out)
    api = f"{args.mirror}/api/models/{REPO}"
    print(f"Querying model tree from {api} ...")
    tree = fetch_json(api)
    files = [s["rfilename"] for s in tree.get("siblings", [])]
    keep = [
        "checkpoint/model_index.json",
        "checkpoint/scheduler/scheduler_config.json",
        "checkpoint/denoiser/config.json",
        "checkpoint/denoiser/diffusion_pytorch_model.safetensors",
        "checkpoint/denoiser_garment/config.json",
        "checkpoint/denoiser_garment/diffusion_pytorch_model.safetensors",
        "checkpoint/image_encoder/config.json",
        "checkpoint/image_encoder/model.safetensors",
        "checkpoint/image_encoder/preprocessor_config.json",
        "checkpoint/text_encoder/config.json",
        "checkpoint/text_encoder/model.safetensors",
        "checkpoint/text_encoder_2/config.json",
        "checkpoint/text_encoder_2/model.safetensors",
        "checkpoint/tokenizer/merges.txt",
        "checkpoint/tokenizer/special_tokens_map.json",
        "checkpoint/tokenizer/tokenizer_config.json",
        "checkpoint/tokenizer/vocab.json",
        "checkpoint/tokenizer_2/merges.txt",
        "checkpoint/tokenizer_2/special_tokens_map.json",
        "checkpoint/tokenizer_2/tokenizer_config.json",
        "checkpoint/tokenizer_2/vocab.json",
        "checkpoint/vae/config.json",
        "checkpoint/vae/diffusion_pytorch_model.safetensors",
        "checkpoint/vae_decoder/decoder.json",
        "checkpoint/vae_decoder/decoder.pt",
    ]
    missing = [f for f in keep if f not in files]
    if missing:
        print("Missing from remote tree:", missing, file=sys.stderr)
        sys.exit(1)

    for rel in keep:
        url = f"{args.mirror}/{REPO}/resolve/main/{rel}"
        download(url, out / rel)

    print("\nAll weights downloaded to", out.resolve())


if __name__ == "__main__":
    main()
