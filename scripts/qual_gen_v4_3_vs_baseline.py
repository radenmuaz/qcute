"""Side-by-side qualitative generation: a qcute.qcute_refine_v4_3 checkpoint vs. a
qcute.bytelm baseline checkpoint, on the SAME prompts (train and val), so the two can be
read directly against each other rather than comparing separate log files.

    uv run python scripts/qual_gen_v4_3_vs_baseline.py \\
        --v4_3_checkpoint checkpoints/qcute_refine_v4_3_l2_k1/best.pt \\
        --baseline_checkpoint checkpoints/bytelm_xs1_ctx32/best.pt \\
        --n_prompts 5 --prompt_bytes 64 --gen_bytes 64 \\
        --out /tmp/v4_3_vs_baseline.txt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qcute.qcute_refine_v4_3 import Config as V4_3Config
from qcute.qcute_refine_v4_3 import RefineLM, generate_no_cache as v4_3_generate_no_cache
from qcute.bytelm import LMConfig, ByteLM, generate_no_cache as baseline_generate_no_cache
from qcute.qcute_refine_v4_3 import load_enwik8, split_train_val


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v4_3_checkpoint", type=Path, required=True)
    p.add_argument("--baseline_checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--n_bytes", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--n_prompts", type=int, default=5, help="prompts drawn from EACH of train/val")
    p.add_argument("--prompt_bytes", type=int, default=64)
    p.add_argument("--gen_bytes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    v4_3_ckpt = torch.load(args.v4_3_checkpoint, map_location=device)
    v4_3_cfg = V4_3Config(**v4_3_ckpt["cfg"])
    v4_3_model = RefineLM(v4_3_cfg).to(device)
    v4_3_model.load_state_dict(v4_3_ckpt["model"])
    v4_3_model.eval()

    baseline_ckpt = torch.load(args.baseline_checkpoint, map_location=device)
    baseline_cfg = LMConfig(**baseline_ckpt["cfg"])
    baseline_model = ByteLM(baseline_cfg).to(device)
    baseline_model.load_state_dict(baseline_ckpt["model"])
    baseline_model.eval()

    print(f"v4_3: {args.v4_3_checkpoint}  step={v4_3_ckpt.get('step')}  Ks={v4_3_cfg.Ks}")
    print(f"baseline: {args.baseline_checkpoint}  step={baseline_ckpt.get('step')}  "
          f"d_model={baseline_cfg.d_model} n_layers={baseline_cfg.n_layers} context={baseline_cfg.context}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)

    lines = []

    def log(s: str):
        print(s)
        lines.append(s)

    torch.manual_seed(args.seed)
    total_len = args.prompt_bytes + args.gen_bytes
    for label, src in (("train", train_data), ("val", val_data)):
        for i in range(args.n_prompts):
            start = torch.randint(0, max(1, len(src) - total_len), (1,)).item()
            window = src[start: start + total_len]
            prompt = window[: args.prompt_bytes]
            ground_truth = window[args.prompt_bytes:]

            v4_3_out = v4_3_generate_no_cache(v4_3_model, prompt, args.gen_bytes, device)
            v4_3_gen = bytes(v4_3_out[prompt.numel():].tolist())

            baseline_out = baseline_generate_no_cache(baseline_model, prompt, args.gen_bytes, device)
            baseline_gen = bytes(baseline_out[prompt.numel():].tolist())

            log(f"=== {label} sample {i} ===")
            log(f"prompt:       {bytes(prompt.tolist())!r}")
            log(f"v4_3:         {v4_3_gen!r}")
            log(f"baseline:     {baseline_gen!r}")
            log(f"ground_truth: {bytes(ground_truth.tolist())!r}")
            log("")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
