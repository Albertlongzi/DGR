#!/usr/bin/env python3
"""Strip a DGR training checkpoint down to inference-only weights for release.

Training checkpoints carry optimizer state, which roughly triples their size and has no
use at inference. This writes ``model.safetensors`` plus a ``config.json`` recording the
architecture hyperparameters needed to rebuild the network.

    python tools/export_checkpoint.py --ckpt runs/stage2/diff_epoch_092.pt \
        --out_dir hf_export/stage2 --kind diffusion --name stage2_diffusion

    python tools/export_checkpoint.py --ckpt runs/stage1/mageultra_epoch_025.pt \
        --out_dir hf_export/stage1 --kind cnn --name stage1_cnn
"""

import argparse
import json
import os

import torch

# Hyperparameters that describe the *architecture*, and so must survive into config.json.
# Everything else in the training args (data roots, schedules, worker counts) is dropped.
ARCH_KEYS = {
    "diffusion": [
        "radius",
        "num_train_timesteps",
        "t2_cond_channels",
        "t2_contrast_mod",
        "t2_canny_low",
        "t2_canny_high",
        "cnn_base_channels",
        "cnn_latent_dim",
        "cnn_prompt_k",
        "cnn_prompt_temp",
        "b_low",
        "b_high",
    ],
    "cnn": [
        "radius",
        "base_channels",
        "latent_dim",
        "prompt_k",
        "prompt_temp",
        "b_low",
        "b_high",
    ],
}

MODEL_CLASS = {
    "diffusion": "dgr.models.diffusion_unet_diffusers.DiffusionUNetT2AndCNN",
    "cnn": "dgr.models.phc_e2e_mageultra_net.PHCE2EMageUltraNet",
}


def _coerce(v):
    """Parse a KEY=VALUE override into int/float/bool/None where that is unambiguous."""
    low = v.lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _clean_state_dict(sd):
    """Drop DDP prefixes and make every tensor a standalone contiguous copy.

    safetensors refuses tensors that share storage, which is exactly what slicing a
    checkpoint out of a fused optimizer buffer produces.
    """
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        if not isinstance(v, torch.Tensor):
            continue
        out[k] = v.detach().cpu().contiguous().clone()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="training checkpoint (.pt)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--kind", required=True, choices=["diffusion", "cnn"])
    ap.add_argument("--name", default=None, help="basename of the exported weights (default: --kind)")
    ap.add_argument("--torch_format", action="store_true", help="also write a plain .pt of the weights")
    ap.add_argument(
        "--arch",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="architecture hyperparameters to record when the checkpoint does not carry them "
        "(stage-1 checkpoints predate argument recording), e.g. --arch radius=2 base_channels=64",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    name = args.name or args.kind

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise SystemExit(f"{args.ckpt}: expected a dict with a 'model' key, got {type(ckpt).__name__}")

    state = _clean_state_dict(ckpt["model"])
    n_params = sum(v.numel() for v in state.values())

    train_args = ckpt.get("args") or {}
    if hasattr(train_args, "__dict__"):
        train_args = vars(train_args)
    config = {
        "model_class": MODEL_CLASS[args.kind],
        "kind": args.kind,
        "source_epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss", ckpt.get("va")),
        "num_parameters": n_params,
        "arch": {k: train_args.get(k) for k in ARCH_KEYS[args.kind] if k in train_args},
    }
    for item in args.arch:
        if "=" not in item:
            raise SystemExit(f"--arch expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        config["arch"][k] = _coerce(v)
    missing = [k for k in ARCH_KEYS[args.kind] if k not in config["arch"]]
    if missing:
        print(f"[warn] architecture keys not recorded in the checkpoint: {', '.join(missing)}")
        print("       pass them with --arch KEY=VALUE so config.json is self-describing.")
    if args.kind == "diffusion":
        config["prediction_type"] = ckpt.get("prediction_type")
        config["noise_scheduler"] = ckpt.get("noise_scheduler_config")

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    try:
        from safetensors.torch import save_file

        dst = os.path.join(args.out_dir, f"{name}.safetensors")
        save_file(state, dst, metadata={"format": "pt"})
    except ImportError:
        dst = os.path.join(args.out_dir, f"{name}.pt")
        torch.save(state, dst)

    if args.torch_format and dst.endswith(".safetensors"):
        torch.save(state, os.path.join(args.out_dir, f"{name}.pt"))

    src_mb = os.path.getsize(args.ckpt) / 1e6
    dst_mb = os.path.getsize(dst) / 1e6
    print(f"{args.ckpt}  ({src_mb:.0f} MB)")
    print(f"  -> {dst}  ({dst_mb:.0f} MB, {n_params/1e6:.1f}M params, epoch {config['source_epoch']})")
    print(f"  -> {os.path.join(args.out_dir, 'config.json')}")


if __name__ == "__main__":
    main()
