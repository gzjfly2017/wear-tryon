#!/usr/bin/env python3
"""End-to-end sanity check of the Mobile-VTON pipeline (PyTorch side).

Runs the exact same tensor flow the Swift app will execute, and verifies that
the combined denoiser (garment UNet + try-on UNet) can be torch.jit.traced —
the operation the CoreML conversion script relies on.

Usage (needs weights downloaded and repo cloned):
    python scripts/verify_pipeline.py \
        --checkpoint Models/mobile-vton/checkpoint \
        --repo Models/Mobile-VTON-repo \
        [--resolution 512]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--resolution", type=int, default=512)
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo))
    import torch
    from transformers import CLIPTextModelWithProjection, AutoModel
    from diffusers import AutoencoderKL
    from Mobile_VTON.models.unets.unet_2d_condition_tryon import UNet2DConditionModel as Unet_Tryon
    from Mobile_VTON.models.unets.unet_2d_condition_garment import UNet2DConditionModel as Unet_Garment
    from Mobile_VTON.models.autoencoders.vae import Decoder

    ckpt = Path(args.checkpoint)
    res = args.resolution
    lat = res // 8

    te1 = CLIPTextModelWithProjection.from_pretrained(ckpt, subfolder="text_encoder").eval()
    te2 = CLIPTextModelWithProjection.from_pretrained(ckpt, subfolder="text_encoder_2").eval()
    ie = AutoModel.from_pretrained(ckpt, subfolder="image_encoder").eval()
    vae = AutoencoderKL.from_pretrained(ckpt, subfolder="vae").eval()
    with open(ckpt / "vae_decoder" / "decoder.json") as f:
        vd_cfg = json.load(f)
    vd = Decoder(**vd_cfg)
    vd.load_state_dict(torch.load(ckpt / "vae_decoder" / "decoder.pt", map_location="cpu"), strict=True)
    vd.eval()
    denoiser = Unet_Tryon.from_pretrained(ckpt, subfolder="denoiser").eval()
    denoiser_garment = Unet_Garment.from_pretrained(ckpt, subfolder="denoiser_garment").eval()
    denoiser.set_sample_size(lat)
    denoiser_garment.set_sample_size(lat)

    class CombinedDenoiser(torch.nn.Module):
        def __init__(self, main_net, garment_net):
            super().__init__()
            self.main = main_net
            self.garment = garment_net
            self.image_proj = main_net.encoder_hid_proj
        def forward(self, person_latent, cloth_latent, sigma, text_embeds,
                    cloth_text_embeds, dinov2_hidden_states):
            image_embeds = self.image_proj(dinov2_hidden_states)
            _, garment_features = self.garment(
                sample=cloth_latent, timestep=sigma,
                encoder_hidden_states=cloth_text_embeds, return_dict=False)
            combo = torch.cat([person_latent, cloth_latent], dim=1)
            return self.main(
                sample=combo, timestep=sigma,
                encoder_hidden_states=text_embeds, return_dict=False,
                garment_features=garment_features,
                added_cond_kwargs={"image_embeds": image_embeds})[0]

    combined = CombinedDenoiser(denoiser, denoiser_garment).eval()

    with torch.no_grad():
        ids = torch.randint(0, 49407, (1, 77)).long()
        h1 = te1(ids, output_hidden_states=True).hidden_states[-2]
        h2 = te2(ids, output_hidden_states=True).hidden_states[-2]
        clip = torch.cat([h1, h2], dim=-1)
        padded = torch.nn.functional.pad(clip, (0, 4096 - clip.shape[-1]))
        prompt_embeds = torch.cat([padded, torch.zeros(1, 256, 4096)], dim=-2)
        assert tuple(prompt_embeds.shape) == (1, 333, 4096), prompt_embeds.shape

        dino = ie(torch.rand(1, 3, 518, 518), output_hidden_states=True).hidden_states[-2]
        assert tuple(dino.shape) == (1, 1370, 768), dino.shape

        z1 = (vae.encode(torch.rand(1, 3, res, res)).latent_dist.mean - 0.0609) * 1.5305
        z2 = (vae.encode(torch.rand(1, 3, res, res)).latent_dist.mean - 0.0609) * 1.5305
        assert tuple(z1.shape) == (1, 16, lat, lat), z1.shape

        sigma = torch.tensor([1.0])
        v = combined(z1, z2, sigma, prompt_embeds, prompt_embeds, dino)
        assert tuple(v.shape) == (1, 16, lat, lat), v.shape

        traced = torch.jit.trace(combined, (z1, z2, sigma, prompt_embeds, prompt_embeds, dino))
        out2 = traced(z1, z2, sigma, prompt_embeds, prompt_embeds, dino)
        assert torch.allclose(v, out2, atol=1e-4), "trace mismatch"

        out_img = vd(v)
        assert tuple(out_img.shape) == (1, 3, res, res), out_img.shape

    print("E2E SANITY PASSED (all shapes verified, trace consistent)")


if __name__ == "__main__":
    main()
