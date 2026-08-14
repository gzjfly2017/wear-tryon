#!/usr/bin/env python3
"""Convert Mobile-VTON PyTorch checkpoint to CoreML models for iOS deployment.

IMPORTANT: This script is designed to run on macOS (GitHub Actions macOS runner),
because coremltools needs native mil storage libs for saving .mlpackage and we
compile .mlpackage -> .mlmodelc using `xcrun coremlcompiler` (macOS only).

Model decomposition (each component becomes one CoreML model, Swift orchestrates):

  text_encoder.mlmodelc      CLIP-1: input_ids [1,77] int32 -> hidden_states[-2] [1,77,768]
  text_encoder_2.mlmodelc    CLIP-2: same
  image_encoder.mlmodelc     DINOv2: image [1,3,518,518] float32 (0-1, ImageNet norm) -> image_embeds
  vae.mlmodelc               person/cloth image [1,3,H,W] (0-1) -> latent (scaled)
  vae_decoder.mlmodelc       latent -> image
  denoiser.mlmodelc          COMBINED try-on denoiser:
                             inputs:
                               person_latent [1,16,lat,lat]
                               cloth_latent  [1,16,lat,lat]
                               sigma         [1]
                               text_embeds   [1,333,4096]
                               cloth_text_embeds [1,333,4096]
                               image_embeds  [1,768]
                             output: velocity [1,16,lat,lat]

The combined denoiser internally runs denoiser_garment (with cloth_text_embeds),
concatenates garment features, then runs the main UNet with person latent +
cloth latent concat (32 channels) — mirroring pipeline `__call__` exactly.

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

    inputs = [ct.TensorType(name=nm, shape=sh) for nm, sh in input_names_shapes]
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
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModelWithProjection, AutoModel

    from Mobile_VTON.models.autoencoders.vae import Decoder
    from Mobile_VTON.models.unets.unet_2d_condition_tryon import UNet2DConditionModel as Unet_Tryon
    from Mobile_VTON.models.unets.unet_2d_condition_garment import UNet2DConditionModel as Unet_Garment

    res = args.resolution
    lat = res // 8

    print("=== Loading checkpoints ===")
    te1 = CLIPTextModelWithProjection.from_pretrained(ckpt, subfolder="text_encoder")
    te2 = CLIPTextModelWithProjection.from_pretrained(ckpt, subfolder="text_encoder_2")
    image_encoder = AutoModel.from_pretrained(ckpt, subfolder="image_encoder")
    vae = AutoencoderKL.from_pretrained(ckpt, subfolder="vae")
    with open(ckpt / "vae_decoder" / "decoder.json") as f:
        vd_cfg = json.load(f)
    vae_decoder = Decoder(**vd_cfg)
    vae_decoder.load_state_dict(
        torch.load(ckpt / "vae_decoder" / "decoder.pt", map_location="cpu"), strict=True
    )
    denoiser = Unet_Tryon.from_pretrained(ckpt, subfolder="denoiser")
    denoiser_garment = Unet_Garment.from_pretrained(ckpt, subfolder="denoiser_garment")

    # 重要:按目标分辨率重算 rope 位置编码(默认 sample_size=128 对应 1024px,
    # 低分辨率推理必须调用 set_sample_size 重新生成 rope buffer)
    lat = res // 8
    denoiser.set_sample_size(lat)
    denoiser_garment.set_sample_size(lat)
    print(f"  sample_size set to {lat} (latent, {res}px)")

    latent_channels = getattr(vae.config, "latent_channels", 16)
    print(f"  latent_channels = {latent_channels}")

    # ---------------------------------------------------------------
    print("=== Converting text encoders (output hidden_states[-2]) ===")
    ids = torch.randint(0, 49407, (1, 77)).long()

    class ClipWrapper(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, input_ids):
            out = self.net(input_ids, output_hidden_states=True)
            return out.hidden_states[-2]

    convert_to_coreml(
        ClipWrapper(te1), (ids,),
        [("input_ids", [1, 77])],
        ["hidden_states"], "text_encoder", out,
    )
    convert_to_coreml(
        ClipWrapper(te2), (ids,),
        [("input_ids", [1, 77])],
        ["hidden_states"], "text_encoder_2", out,
    )

    # ---------------------------------------------------------------
    print("=== Converting image encoder (DINOv2) ===")
    # pipeline: image_embeds = image_encoder(image, output_hidden_states=True).hidden_states[-2]
    # 然后 denoiser.encoder_hid_proj(Resampler) 投影到 [1,16,4096]。
    # 此处 DINOv2 输出 hidden_states[-2] ([1,1370,768]);
    # encoder_hid_proj 在合并 denoiser 内部完成(见 CombinedDenoiser)。
    img = torch.rand(1, 3, 518, 518)

    class Dinov2Wrapper(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, image):
            out = self.net(image, output_hidden_states=True)
            return out.hidden_states[-2]  # [1,1370,768]

    convert_to_coreml(
        Dinov2Wrapper(image_encoder), (img,),
        [("image", [1, 3, 518, 518])],
        ["hidden_states"], "image_encoder", out,
    )

    # ---------------------------------------------------------------
    print("=== Converting VAE encoder ===")
    class VAEEncodeWrapper(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.shift = 0.0609
            self.scale = 1.5305
        def forward(self, image):
            dist = self.net.encode(image).latent_dist
            latent = dist.mean
            return (latent - self.shift) * self.scale

    convert_to_coreml(
        VAEEncodeWrapper(vae), (img,),
        [("image", [1, 3, res, res])],
        ["latent"], "vae", out,
    )

    # ---------------------------------------------------------------
    print("=== Converting VAE decoder ===")
    latent = torch.rand(1, latent_channels, lat, lat)
    convert_to_coreml(
        vae_decoder, (latent,),
        [("latent", [1, latent_channels, lat, lat])],
        ["image"], "vae_decoder", out,
    )

    # ---------------------------------------------------------------
    print("=== Converting combined denoiser ===")
    # 与 pipeline 完全一致的合并推理:
    #   gen_condition = vae.encode(person) -> [1,16,h,w] (已由 Swift 端完成)
    #   cloth latent 由 Swift 端完成
    #   组合 latent = cat([person, cloth], dim=1) -> [1,32,h,w]
    #   image_embeds = encoder_hid_proj(dinov2.hidden_states[-2]) -> [1,16,4096]
    #     (投影由 Swift 端预先完成:image_encoder_hid_proj.mlmodelc 或此处包装)
    #   garment_features = denoiser_garment(cloth_latent, sigma, cloth_text_embeds)
    #   velocity = denoiser(combo_latent, sigma, text_embeds, garment_features, image_embeds)
    #     模型内部 ip_image_proj 分支: cat([text_proj(text), image_embeds], dim=1)
    class CombinedDenoiser(torch.nn.Module):
        def __init__(self, main_net, garment_net):
            super().__init__()
            self.main = main_net
            self.garment = garment_net
            # 将 Resampler 投影独立暴露,便于 Swift 端单独调用
            self.image_proj = main_net.encoder_hid_proj
        def forward(self, person_latent, cloth_latent, sigma, text_embeds,
                    cloth_text_embeds, dinov2_hidden_states):
            # image_embeds: Resampler 投影(与 pipeline prepare_ip_adapter_image_embeds 一致)
            image_embeds = self.image_proj(dinov2_hidden_states)
            # garment features
            _, garment_features = self.garment(
                sample=cloth_latent,
                timestep=sigma,
                encoder_hidden_states=cloth_text_embeds,
                return_dict=False,
            )
            # combined latent: person + cloth (32ch)
            combo = torch.cat([person_latent, cloth_latent], dim=1)
            velocity = self.main(
                sample=combo,
                timestep=sigma,
                encoder_hidden_states=text_embeds,
                return_dict=False,
                garment_features=garment_features,
                added_cond_kwargs={"image_embeds": image_embeds},
            )[0]
            return velocity

    combined = CombinedDenoiser(denoiser, denoiser_garment).eval()
    # text_embeds: [1,333,4096] (77 clip pad + 256 zero t5), cloth same
    te_embeds = torch.rand(1, 333, 4096)
    person_lat = torch.rand(1, latent_channels, lat, lat)
    cloth_lat = torch.rand(1, latent_channels, lat, lat)
    sigma = torch.tensor([1.0])
    dino_hs = torch.rand(1, 1370, 768)  # DINOv2 hidden_states[-2]
    convert_to_coreml(
        combined,
        (person_lat, cloth_lat, sigma, te_embeds, te_embeds, dino_hs),
        [
            ("person_latent", [1, latent_channels, lat, lat]),
            ("cloth_latent", [1, latent_channels, lat, lat]),
            ("sigma", [1]),
            ("text_embeds", [1, 333, 4096]),
            ("cloth_text_embeds", [1, 333, 4096]),
            ("dinov2_hidden_states", [1, 1370, 768]),
        ],
        ["velocity"], "denoiser", out,
    )

    print(f"\nAll CoreML models written to {out.resolve()}")
    print("Copy *.mlmodelc into WearTryOn/Resources/Models before building.")


if __name__ == "__main__":
    main()
