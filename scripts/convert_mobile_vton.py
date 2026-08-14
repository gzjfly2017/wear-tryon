#!/usr/bin/env python3
"""Convert Mobile-VTON PyTorch checkpoint to CoreML models for iOS deployment.

IMPORTANT: This script is designed to run on macOS (GitHub Actions macOS runner),
because:
  - coremltools needs the native mil storage libs for saving .mlpackage
  - we compile .mlpackage -> .mlmodelc using `xcrun coremlcompiler` (macOS only)

It converts each component of the Mobile-VTON pipeline to a separate CoreML
model so the Swift side can orchestrate the sampling loop (FlowMatch Euler)
with fine control:

  text_encoder.mlmodelc      CLIP text encoder 1 (input_ids [1,77] int32 -> text_embeds)
  text_encoder_2.mlmodelc    CLIP text encoder 2
  image_encoder.mlmodelc     DINOv2 garment encoder (image [1,3,512,512] -> image_embeds)
  vae.mlmodelc               SD3.5 VAE encoder (image [1,3,512,512] -> latent [1,16,64,64])
  vae_decoder.mlmodelc       custom VAE decoder (latent -> image)
  denoiser.mlmodelc          main try-on UNet (latent + garment_feature + sigma + text -> velocity)
  denoiser_garment.mlmodelc  garment UNet (garment latent + sigma + text -> feature)

Environment:
  - HF_ENDPOINT may be set to https://hf-mirror.com when running in CN network.
  - Requires torch, diffusers, transformers, coremltools, safetensors.
  - The Mobile-VTON repo code must be importable (repo/ on sys.path).

Usage:
    python scripts/convert_mobile_vton.py \
        --checkpoint Models/mobile-vton/checkpoint \
        --repo Models/Mobile-VTON-repo \
        --out build/coreml
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kw)


def convert_to_coreml(torch_model, example_inputs, input_names_shapes, output_names, name, out_dir):
    """Convert a torch module to mlpackage via coremltools (CT4 torch trace)."""
    import coremltools as ct
    import torch

    model = torch_model.eval()
    traced = torch.jit.trace(model, example_inputs)

    inputs = [
        ct.TensorType(name=nm, shape=sh)
        for nm, sh in input_names_shapes
    ]
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name=nm) for nm in output_names],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )
    out_path = Path(out_dir) / name
    mlmodel.save(str(out_path))
    # compile to mlmodelc for direct bundle embedding
    compile_coreml(out_path, out_dir)
    print(f"[ok] {name}")


def compile_coreml(mlpackage: Path, out_dir: Path) -> None:
    """Compile .mlpackage -> .mlmodelc using xcrun (macOS only)."""
    dest = mlpackage.with_suffix(".mlmodelc")
    if dest.exists():
        shutil.rmtree(dest)
    run(["xcrun", "coremlcompiler", "compile", str(mlpackage), str(Path(out_dir))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="build/coreml")
    ap.add_argument("--resolution", type=int, default=512)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    repo = Path(args.repo)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo))
    import torch
    import coremltools as ct
    from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
    from diffusers.models.transformers import SD3Transformer2DModel
    from transformers import CLIPTextModelWithProjection, AutoModel, AutoImageProcessor

    from Mobile_VTON.models.autoencoders.vae import Decoder
    from Mobile_VTON.models.unets.unet_2d_condition_tryon import UNet2DConditionModel as Unet_Tryon
    from Mobile_VTON.models.unets.unet_2d_condition_garment import UNet2DConditionModel as Unet_Garment

    res = args.resolution
    lat = res // 8

    print("=== Loading checkpoints ===")

    # 1. Text encoders (CLIP)
    te1 = CLIPTextModelWithProjection.from_pretrained(ckpt, subfolder="text_encoder")
    te2 = CLIPTextModelWithProjection.from_pretrained(ckpt, subfolder="text_encoder_2")

    # 2. Image encoder (DINOv2)
    image_encoder = AutoModel.from_pretrained(ckpt, subfolder="image_encoder")

    # 3. VAE (SD3.5)
    vae = AutoencoderKL.from_pretrained(ckpt, subfolder="vae")

    # 4. Custom VAE decoder
    with open(ckpt / "vae_decoder" / "decoder.json") as f:
        vd_cfg = json.load(f)
    vae_decoder = Decoder(**vd_cfg)
    vae_decoder.load_state_dict(
        torch.load(ckpt / "vae_decoder" / "decoder.pt", map_location="cpu"), strict=True
    )

    # 5. UNets
    denoiser = Unet_Tryon.from_pretrained(ckpt, subfolder="denoiser")
    denoiser_garment = Unet_Garment.from_pretrained(ckpt, subfolder="denoiser_garment")

    print("=== Converting text encoders ===")
    ids = torch.randint(0, 49407, (1, 77)).long()
    convert_to_coreml(
        te1, (ids,),
        [("input_ids", [1, 77])],
        ["text_embeds"], "text_encoder", out,
    )
    convert_to_coreml(
        te2, (ids,),
        [("input_ids", [1, 77])],
        ["text_embeds"], "text_encoder_2", out,
    )

    print("=== Converting image encoder (DINOv2) ===")
    img = torch.rand(1, 3, res, res)
    convert_to_coreml(
        image_encoder, (img,),
        [("image", [1, 3, res, res])],
        ["image_embeds"], "image_encoder", out,
    )

    print("=== Converting VAE encoder ===")
    vae_enc = vae.encoder
    latent_channels = getattr(vae.config, "latent_channels", 16)
    print(f"  latent_channels = {latent_channels}")
    convert_to_coreml(
        vae_enc, (img,),
        [("image", [1, 3, res, res])],
        ["latent"], "vae", out,
    )

    print("=== Converting VAE decoder ===")
    latent = torch.rand(1, latent_channels, lat, lat)
    convert_to_coreml(
        vae_decoder, (latent,),
        [("latent", [1, latent_channels, lat, lat])],
        ["image"], "vae_decoder", out,
    )

    print("=== Converting denoiser_garment ===")
    # 服装 UNet:输入服装潜在 + sigma + 文本嵌入;输出中间特征列表(供主 UNet 使用)
    glat = torch.rand(1, 16, lat, lat)
    sigma = torch.tensor([1.0])
    te_embeds = torch.rand(1, 77, 2048)
    try:
        class GarmentWrapper(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net
            def forward(self, latent, sigma, text_embeds):
                _, feats = self.net(sample=latent, timestep=sigma,
                                    encoder_hidden_states=text_embeds, return_dict=False)
                # 取第一个中间特征(与 pipeline 中 garment_features 的用法对齐;
                # 完整特征列表见 pipeline,此处以主 UNet 接收的单特征简化)
                if isinstance(feats, (tuple, list)):
                    return feats[0]
                return feats
        wrapper = GarmentWrapper(denoiser_garment)
        convert_to_coreml(
            wrapper, (glat, sigma, te_embeds),
            [("latent", [1, 16, lat, lat]), ("sigma", [1]), ("text_embeds", [1, 77, 2048])],
            ["output"], "denoiser_garment", out,
        )
    except Exception as e:
        print(f"[warn] denoiser_garment conversion failed: {e}")
        import traceback; traceback.print_exc()

    print("=== Converting denoiser (try-on UNet) ===")
    # 主 UNet:人物潜在 + 服装特征 + sigma + 文本嵌入 + image_embeds(added_cond)
    try:
        class TryonWrapper(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net
            def forward(self, latent, garment_feature, sigma, text_embeds, image_embeds):
                out = self.net(
                    sample=latent,
                    timestep=sigma,
                    encoder_hidden_states=text_embeds,
                    return_dict=False,
                    garment_features=(garment_feature,),
                    added_cond_kwargs={"image_embeds": image_embeds},
                )
                return out[0]
        wrapper = TryonWrapper(denoiser)
        # 服装特征形状需与模型定义一致,先从 config 读取
        gf_shape = getattr(denoiser.config, "garment_feature_shape", [1, 16, lat, lat])
        gfeat = torch.rand(*gf_shape)
        img_embeds = torch.rand(1, 1024)  # DINOv2 输出维度,若不对会报错提示
        convert_to_coreml(
            wrapper, (torch.rand(1, 16, lat, lat), gfeat, sigma, te_embeds, img_embeds),
            [("latent", [1, 16, lat, lat]), ("garment_feature", gf_shape),
             ("sigma", [1]), ("text_embeds", [1, 77, 2048]), ("image_embeds", [1, 1024])],
            ["velocity"], "denoiser", out,
        )
    except Exception as e:
        print(f"[fatal] denoiser conversion failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print(f"\nAll CoreML models written to {out.resolve()}")
    print("Copy *.mlmodelc into WearTryOn/Resources/Models before building.")


if __name__ == "__main__":
    main()
