"""scripts/qual_gen_bytelm.py — simple qualitative generation for
qcute.bytelm baselines, same sample-picking convention as
scripts/qual_gen_v4_2.py (same seed/offset logic, so both scripts draw
the SAME prompts from the SAME data when run with matching --seed/
--prompt_bytes/--gen_bytes/--n_train/--n_val, for an apples-to-apples
comparison against the qcute_refine_v4_2 qualitative output).

Session: "for sample inputs, sample from bytelm baseline xs3 and bytelm
4 layer" (checkpoints/bytelm_xs3_ctx1024 -- n_layers=3, and
checkpoints/bytelm_xs_mtp4_ctx1024 -- n_layers=4, both context=1024,
"1024 model") "output different files" -- one .txt per checkpoint, not
combined.

Greedy decode via generate_no_cache (deterministic, matches the
"obviously correct" reference used elsewhere in this codebase; no KV-cache
speed concern needed at this scale/sample count).

    uv run python scripts/qual_gen_bytelm.py
    uv run python scripts/qual_gen_bytelm.py --checkpoints checkpoints/bytelm_xs3_ctx1024/best.pt checkpoints/bytelm_xs_mtp4_ctx1024/best.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qcute.bytelm import LMConfig, ByteLM, generate_no_cache
from qcute.qcute_refine_v4_2 import load_enwik8, split_train_val


def pick_offsets(region_len: int, n: int, prompt_len: int, gen_len: int, align: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    max_start = region_len - (prompt_len + gen_len)
    if max_start <= 0:
        raise SystemExit(f"region too short ({region_len} bytes) for prompt_len={prompt_len} + gen_len={gen_len}")
    offsets = (torch.rand(n, generator=g) * max_start).long()
    return sorted((o - o % align).item() for o in offsets)


def b2s(b: bytes) -> str:
    return b.decode("latin-1").translate({9: " ", 10: "↵", 13: ""})


def main():
    p = argparse.ArgumentParser(description="Qualitative AR generation for qcute.bytelm baseline checkpoints")
    p.add_argument("--checkpoints", type=Path, nargs="+",
                    default=[Path("checkpoints/bytelm_xs3_ctx1024/best.pt"), Path("checkpoints/bytelm_xs_mtp4_ctx1024/best.pt")],
                    help="one output file per checkpoint given")
    p.add_argument("--n_train", type=int, default=10)
    p.add_argument("--n_val", type=int, default=2)
    p.add_argument("--prompt_bytes", type=int, default=64)
    p.add_argument("--gen_bytes", type=int, default=64)
    p.add_argument("--align", type=int, default=32, help="matches qual_gen_v4_2.py's own Ks[0]=32 default, so offsets line up across scripts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--out_dir", type=Path, default=Path("logs/qual_gen_bytelm"))
    p.add_argument("--device", default=None, help="override auto device pick -- e.g. 'cpu' to avoid contending with a concurrent MPS training job")
    args = p.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    data = load_enwik8(args.data)
    train_data, val_data = split_train_val(data, args.val_frac)

    samples = []
    for region_name, region, n in [("train", train_data, args.n_train), ("val", val_data, args.n_val)]:
        for off in pick_offsets(len(region), n, args.prompt_bytes, args.gen_bytes, args.align, args.seed):
            samples.append((region_name, off, region))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for ckpt_path in args.checkpoints:
        run_name = ckpt_path.parent.name
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = LMConfig(**ckpt["cfg"])
        model = ByteLM(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"\n=== {run_name} ===")
        print(f"loaded {ckpt_path} (step={ckpt.get('step')})  n_layers={cfg.n_layers} d_model={cfg.d_model} context={cfg.context} mtp_heads={cfg.mtp_heads}\n")

        txt_path = args.out_dir / f"{run_name}_{stamp}.txt"
        lines = [f"model: {run_name}\ncheckpoint: {ckpt_path}\nn_layers={cfg.n_layers} d_model={cfg.d_model} context={cfg.context}\n\n"]

        for region_name, off, region in samples:
            prompt = region[off: off + args.prompt_bytes]
            gt = region[off + args.prompt_bytes: off + args.prompt_bytes + args.gen_bytes]
            out = generate_no_cache(model, prompt, args.gen_bytes, device)
            gen_cont = out[args.prompt_bytes:].cpu()

            block = (
                f"=== {region_name} offset={off} ===\n"
                f"prompt      : {b2s(bytes(prompt.tolist()))}\n"
                f"gen (cont.) : {b2s(bytes(gen_cont.tolist()))}\n"
                f"gt  (cont.) : {b2s(bytes(gt.tolist()))}\n\n"
            )
            print(block, end="")
            lines.append(block)

        txt_path.write_text("".join(lines))
        print(f"wrote {txt_path}")


if __name__ == "__main__":
    main()
