"""Greedy (argmax) qualitative generation for a saved qcute_refine_v4_4 or v4_5 checkpoint, from a
FIXED prompt offset (unlike qualitative_generate's own random window) -- for directly comparable
samples across architectures/checkpoints on the identical prompt.

    uv run python scripts/refine_greedy_qual.py --module qcute_refine_v4_4 \
        --checkpoint checkpoints/qcute_refine_v4_4_overfit10k_k4_1_l1/best.pt \
        --data datasets/enwik8_1M.gz --n_bytes 10000 --qual_source train --gen_len 64 --prompt_len 64 --start 5850
"""
import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--module", choices=["qcute_refine_v4_4", "qcute_refine_v4_5"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--n_bytes", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--qual_source", choices=["train", "val"], default="train")
    p.add_argument("--gen_len", type=int, default=64)
    p.add_argument("--prompt_len", type=int, default=64)
    p.add_argument("--start", type=int, default=None)
    args = p.parse_args()

    mod = importlib.import_module(f"qcute.{args.module}")
    device = "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = mod.Config(**ckpt["cfg"])
    model = mod.RefineLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data = mod.load_enwik8(args.data, args.n_bytes)
    train_data, val_data = mod.split_train_val(data, args.val_frac)
    src = train_data if args.qual_source == "train" else val_data

    total_len = args.prompt_len + args.gen_len
    start = args.start if args.start is not None else torch.randint(0, max(1, len(src) - total_len), (1,)).item()
    window = src[start: start + total_len]
    prompt = window[: args.prompt_len]
    ground_truth = window[args.prompt_len:]

    print(f"module={args.module} qual_source={args.qual_source} step={ckpt['step']} start={start}")
    mod.qualitative_generate(model, prompt, args.gen_len, ground_truth, device, log=print, label=args.qual_source)


if __name__ == "__main__":
    main()
