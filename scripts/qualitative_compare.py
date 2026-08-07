"""Generate continuations from the same prompts across all three trained
baselines (qcute.bytelm, qcute.bpelm, qcute.qcutelm) and save prompt/
generated/ground-truth side by side, for a qualitative "is it garbage or
plausible English" read — a complement to the aggregate bpb numbers, which
say nothing about what the text actually looks like.

Prompts are drawn as raw byte slices from datasets/enwik8_1M.gz — some
from the train region, some from the val region (same byte-level split as
all three training scripts use) — so "same warmup" means literally the same
underlying bytes fed to every model, not just the same nominal prompt
length. Ground truth is the real corpus bytes right after the prompt.

    uv run python scripts/qualitative_compare.py \\
        --bytelm_checkpoint checkpoints/bytelm_xs_mtp4/best.pt \\
        --bpelm_checkpoint checkpoints/bpelm_8192/best.pt \\
        --bpelm_sp_model datasets/bpe_enwik8_1M_8192.model \\
        --qcutelm_checkpoint checkpoints/qcutelm_bsq_k4_lfq_aux/best.pt

Writes both a human-readable .txt and a .json to --out_dir (default
logs/qualitative_compare/).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sentencepiece as spm
import torch

from qcute.bytelm import LMConfig, ByteLM, generate_speculative, score_continuation_bpb as bytelm_bpb
from qcute.bpelm import (
    BpeLMConfig, BpeLM, build_byte_len_table, generate_ar as bpelm_generate_ar,
    score_continuation_bpb as bpelm_bpb,
)
from qcute.archive.qcutelm import Config as QCuteConfig, QCuteLM, load_enwik8, split_train_val


def pick_offsets(region_len: int, n: int, prompt_len: int, gen_len: int, align: int, seed: int) -> list[int]:
    """n roughly-evenly-spaced offsets into a region, each rounded down to a
    multiple of `align` (qcutelm's K) so the same offset works for every
    model, leaving room for prompt_len + gen_len bytes after it."""
    g = torch.Generator().manual_seed(seed)
    max_start = region_len - (prompt_len + gen_len)
    if max_start <= 0:
        raise SystemExit(f"region too short ({region_len} bytes) for prompt_len={prompt_len} + gen_len={gen_len}")
    offsets = (torch.rand(n, generator=g) * max_start).long()
    return sorted((o - o % align).item() for o in offsets)


def run_bytelm(checkpoint_path: Path, prompt: bytes, gt: bytes, device: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = LMConfig(**ckpt["cfg"])
    model = ByteLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    prompt_t = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    out, _ = generate_speculative(model, prompt_t, len(gt))
    gen_bytes = bytes(out[0, prompt_t.size(1):].tolist())
    bpb = bytelm_bpb(model, prompt + gt, len(prompt), device)
    return {"generated": gen_bytes, "bpb_on_ground_truth": bpb}


def run_bpelm(checkpoint_path: Path, sp_model: Path, prompt: bytes, gt: bytes, device: str) -> dict:
    sp = spm.SentencePieceProcessor(model_file=str(sp_model))
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = BpeLMConfig(**ckpt["cfg"])
    model = BpeLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    byte_len_table = build_byte_len_table(sp).to(device)

    prompt_text = prompt.decode("utf-8", errors="replace")
    prompt_ids = torch.tensor([sp.encode(prompt_text)], dtype=torch.long, device=device)
    # over-generate tokens (BPE tokens average >1 byte each on this corpus,
    # see scripts/train_bpe.py) then truncate the decoded bytes to len(gt)
    # for a length-matched comparison against the other two models.
    out_ids = bpelm_generate_ar(model, prompt_ids, n_new_tokens=len(gt))
    gen_text = sp.decode(out_ids[0].tolist())
    gen_bytes_full = gen_text.encode("utf-8")
    gen_bytes = gen_bytes_full[len(prompt_text.encode("utf-8")):][: len(gt)]

    full_ids = torch.tensor([sp.encode(prompt_text + gt.decode("utf-8", errors="replace"))], dtype=torch.long, device=device)
    prompt_token_len = len(sp.encode(prompt_text))
    bpb = bpelm_bpb(model, full_ids, prompt_token_len, byte_len_table, device) if full_ids.size(1) > prompt_token_len else float("nan")
    return {"generated": gen_bytes, "bpb_on_ground_truth": bpb}


def run_qcutelm(checkpoint_path: Path, prompt: bytes, gt: bytes, device: str) -> dict:
    from qcute.archive.qcutelm import qualitative_generate, score_continuation_bpb as qcutelm_bpb

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = QCuteConfig(**ckpt["cfg"])
    model = QCuteLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    K = cfg.K
    assert len(prompt) % K == 0 and len(gt) % K == 0, "prompt/gen lengths must be multiples of qcutelm's K"

    prompt_chunks = torch.tensor([list(prompt)], dtype=torch.long, device=device).reshape(1, -1, K)
    gt_chunks = torch.tensor([list(gt)], dtype=torch.long, device=device).reshape(1, -1, K)
    out = model.generate(prompt_chunks, gt_chunks.size(1))
    gen_bytes = bytes(out[0, prompt_chunks.size(1):].reshape(-1).tolist())
    full_chunks = torch.cat([prompt_chunks, gt_chunks], dim=1)
    bpb = qcutelm_bpb(model, full_chunks, prompt_chunks.size(1), device)
    return {"generated": gen_bytes, "bpb_on_ground_truth": bpb}


def main():
    p = argparse.ArgumentParser(description="Qualitative generation comparison: bytelm vs bpelm vs qcutelm")
    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--n_bytes", type=int, default=2_000_000)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--n_prompts_train", type=int, default=3)
    p.add_argument("--n_prompts_val", type=int, default=3)
    p.add_argument("--prompt_bytes", type=int, default=64)
    p.add_argument("--gen_bytes", type=int, default=64)
    p.add_argument("--align", type=int, default=4, help="round prompt offsets to a multiple of this (qcutelm's K)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bytelm_checkpoint", type=Path, required=True)
    p.add_argument("--bpelm_checkpoint", type=Path, required=True)
    p.add_argument("--bpelm_sp_model", type=Path, required=True)
    p.add_argument("--qcutelm_checkpoint", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=Path("logs/qualitative_compare"))
    args = p.parse_args()

    if args.prompt_bytes % args.align or args.gen_bytes % args.align:
        raise SystemExit(f"--prompt_bytes and --gen_bytes must be multiples of --align={args.align}")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)

    prompts = []
    for region_name, region in [("train", train_data), ("val", val_data)]:
        n = args.n_prompts_train if region_name == "train" else args.n_prompts_val
        for off in pick_offsets(len(region), n, args.prompt_bytes, args.gen_bytes, args.align, args.seed):
            prompt = bytes(region[off : off + args.prompt_bytes].tolist())
            gt = bytes(region[off + args.prompt_bytes : off + args.prompt_bytes + args.gen_bytes].tolist())
            prompts.append({"region": region_name, "offset": off, "prompt": prompt, "ground_truth": gt})

    results = []
    for entry in prompts:
        prompt, gt = entry["prompt"], entry["ground_truth"]
        print(f"[{entry['region']} offset={entry['offset']}] generating...")
        row = {"region": entry["region"], "offset": entry["offset"], "prompt": prompt, "ground_truth": gt}
        row["bytelm"] = run_bytelm(args.bytelm_checkpoint, prompt, gt, device)
        row["bpelm"] = run_bpelm(args.bpelm_checkpoint, args.bpelm_sp_model, prompt, gt, device)
        row["qcutelm"] = run_qcutelm(args.qcutelm_checkpoint, prompt, gt, device)
        results.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = args.out_dir / f"compare_{stamp}.json"
    txt_path = args.out_dir / f"compare_{stamp}.txt"

    def b2s(b: bytes) -> str:
        return repr(b)

    with open(json_path, "w") as f:
        json.dump(
            [
                {
                    "region": r["region"], "offset": r["offset"],
                    "prompt": r["prompt"].decode("utf-8", errors="replace"),
                    "ground_truth": r["ground_truth"].decode("utf-8", errors="replace"),
                    **{
                        model: {"generated": r[model]["generated"].decode("utf-8", errors="replace"),
                                "bpb_on_ground_truth": r[model]["bpb_on_ground_truth"]}
                        for model in ("bytelm", "bpelm", "qcutelm")
                    },
                }
                for r in results
            ],
            f, indent=2,
        )

    with open(txt_path, "w") as f:
        for r in results:
            f.write(f"=== {r['region']} offset={r['offset']} ===\n")
            f.write(f"prompt:       {b2s(r['prompt'])}\n")
            f.write(f"ground_truth: {b2s(r['ground_truth'])}\n")
            for model in ("bytelm", "bpelm", "qcutelm"):
                bpb = r[model]["bpb_on_ground_truth"]
                f.write(f"{model:9s}gen: {b2s(r[model]['generated'])}  (bpb_on_gt={bpb:.3f})\n")
            f.write("\n")

    print(f"\nwrote {json_path}\nwrote {txt_path}")


if __name__ == "__main__":
    main()
