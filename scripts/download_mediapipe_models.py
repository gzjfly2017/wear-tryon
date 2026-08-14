#!/usr/bin/env python3
"""Download MediaPipe model files for the iOS app bundle.

Models (from Google AI Edge MediaPipe Tasks model zoo):
  - selfie_segmenter.tflite       : real-time body segmentation
  - pose_landmarker_lite.task     : pose landmark detection (Tasks format)

Uses curl for reliable downloads on both Windows and macOS runners.

Usage:
    python scripts/download_mediapipe_models.py [--out WearTryOn/Resources/Models]
"""
import argparse
import subprocess
import sys
from pathlib import Path

MODELS = {
    "selfie_segmenter.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} (exists)")
        return
    print(f"  [get ] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    r = subprocess.run(
        ["curl", "-sL", "--retry", "3", "--connect-timeout", "30",
         "--max-time", "600", "-o", str(tmp), url],
        check=False,
    )
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        print(f"  [fail] {dest.name}", file=sys.stderr)
        sys.exit(1)
    tmp.rename(dest)
    print(f"  [done] {dest.name} ({dest.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="WearTryOn/Resources/Models")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        download(url, out / name)
    print("All MediaPipe models ready.")


if __name__ == "__main__":
    main()
