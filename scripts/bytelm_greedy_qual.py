"""Greedy (argmax) qualitative generation for a saved bytelm checkpoint -- qcute.bytelm's own
qualitative_generate uses generate_speculative (temperature-sampled), which isn't apples-to-apples
with qcute_refine's generation functions (all pure argmax, no sampling). This script calls
generate_no_cache directly instead, for a fair comparison against v4.4/v4.5 samples.

    uv run python scripts/bytelm_greedy_qual.py --checkpoint checkpoints/bytelm_overfit10k_l1/best.pt \
        --data datasets/enwik8_1M.gz --n_bytes 10000 --qual_source train --gen_len 64 --prompt_len 64
"""
import argparse
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qcute.bytelm import ByteLM, LMConfig, generate_no_cache, split_train_val


def load_enwik8(path: Path, n_bytes: int | None = None) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read(n_bytes) if n_bytes else f.read()
    return torch.tensor(list(data), dtype=torch.long)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--n_bytes", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--qual_source", choices=["train", "val"], default="train")
    p.add_argument("--gen_len", type=int, default=64)
    p.add_argument("--prompt_len", type=int, default=64)
    p.add_argument("--start", type=int, default=None)
    args = p.parse_args()

    device = "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = LMConfig(**ckpt["cfg"])
    model = ByteLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    src = train_data if args.qual_source == "train" else val_data

    total_len = args.prompt_len + args.gen_len
    start = args.start if args.start is not None else torch.randint(0, max(1, len(src) - total_len), (1,)).item()
    window = src[start: start + total_len]
    prompt = window[: args.prompt_len]
    ground_truth = window[args.prompt_len:]

    out = generate_no_cache(model, prompt.unsqueeze(0), args.gen_len, device)
    gen_bytes = bytes(out[prompt.numel():].tolist())

    print(f"qual_source={args.qual_source} step={ckpt['step']}")
    print(f"qual_prompt:       {bytes(prompt.tolist())!r}")
    print(f"qual_generated_greedy: {gen_bytes!r}")
    print(f"qual_ground_truth: {bytes(ground_truth.tolist())!r}")


if __name__ == "__main__":
    main()
