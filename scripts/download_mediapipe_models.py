#!/usr/bin/env python3
"""Download MediaPipe model files for the iOS app bundle.

Models (from Google AI Edge MediaPipe Tasks):
  - selfie_segmenter.tflite     : real-time body segmentation
  - pose_landmarker_lite.tflite : pose landmark detection

The tflite files are copied into WearTryOn/Resources/Models at build time.
We download them from the official google-ai-edge/mediapipe GitHub repo releases.

Usage:
    python scripts/download_mediapipe_models.py [--out WearTryOn/Resources/Models]
"""
import argparse
import urllib.request
from pathlib import Path

# Official release URL pattern:
# https://github.com/google-ai-edge/mediapipe/releases/download/v0.10.14/...
MODELS = {
    "selfie_segmenter.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
    "pose_landmarker_lite.tflite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.tflite",
}


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} (exists)")
        return
    print(f"  [get ] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "wear-tryon/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    tmp.rename(dest)
    print(f"  [done] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


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
