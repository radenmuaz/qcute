"""For n_levels>1 checkpoints, isolate whether cond_full free-running generation's drift (correct
prefix, then diverges -- see docs/status.md) is caused by upstream (level>=1) code quality
compounding, vs. the level-0 decode mechanism itself.

v2: the first version of this script was itself buggy -- it sourced the "oracle" level-1 track
from `encode_lms[1]`'s raw output, but with `cross_track_source="decode"` the real model
conditions on `decode_derived_c[1]` (level 1's own self-decoded/refined output), a different
tensor. That fed decode_lms[0] out-of-distribution conditioning it was never trained on, so the
"oracle" result was meaningless. This version calls `model._run` directly on the ground-truth byte
sequence and reads its real `decode_derived_c[1]` back out (see qcute_v5_concat.py's `_run`, which
now returns it) instead of reimplementing the cross_track_source dispatch by hand -- avoids
repeating the same class of mistake. Every intermediate tensor is printed so this can be stepped
through, not just trusted.

    uv run python scripts/test_v5_concat_cond_full_oracle_level1.py <checkpoint.pt> [--offset N]
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qcute.qcute_v5_concat import (
    Config, RefineLM, generate_no_cache, generate_self_only_cond,
    _sample_next_byte, load_enwik8, split_train_val,
)

START, PROMPT_LEN, GEN_LEN = 5850, 64, 64


def _code_embed(model: "RefineLM", level: int, source_c: torch.Tensor) -> torch.Tensor:
    cfg = model.cfg
    decode_i = model.decode_lms[level]
    if cfg.quant_type == "bsq":
        return decode_i.code_embed(source_c if cfg.decode_code_ste else source_c.detach())
    elif cfg.decode_code_ste:
        return source_c @ decode_i.embed.weight
    else:
        return decode_i.embed(source_c.argmax(-1))


@torch.no_grad()
def generate_cond_full_oracle_coarse(model: "RefineLM", prompt_bytes: torch.Tensor,
                                      full_ground_truth_bytes: torch.Tensor, n_new_bytes: int,
                                      device: str, verbose: bool = False) -> torch.Tensor:
    cfg = model.cfg
    assert model.n_levels == 2, "this script's oracle chain only handles n_levels==2 for now"
    was_training = model.training
    model.eval()

    prompt_bytes = prompt_bytes.to(device)
    full_ground_truth_bytes = full_ground_truth_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    if full_ground_truth_bytes.dim() == 1:
        full_ground_truth_bytes = full_ground_truth_bytes.unsqueeze(0)
    assert full_ground_truth_bytes.shape[1] >= prompt_bytes.shape[1] + n_new_bytes, (
        "full_ground_truth_bytes must cover prompt + all n_new_bytes positions")

    decode_K = 1
    for k in cfg.Ks:
        decode_K *= k

    # Step 1: run the REAL model._run on the ground-truth byte sequence and read back its real
    # decode_derived_c[1] -- this is exactly what cond_full's own cross-attention track would be
    # if levels 0 and 1 both saw perfect (ground-truth) bytes throughout, computed by the actual
    # production code path, not a hand-rolled reimplementation.
    _, _, _, _, _, gt_c_list, _, _, _, gt_decode_derived_c = model._run(
        full_ground_truth_bytes, compute_ntp=False, max_decode_sources=None, want_next_query=False)
    assert 1 in gt_decode_derived_c, (
        "level 1 has no decode_derived_c entry -- check cross_track_source/decode_windows on this "
        "checkpoint's config; the oracle track must come from the same place cond_full uses")
    gt_c1_decode_derived = gt_decode_derived_c[1]
    if verbose:
        print(f"  [oracle] gt_c_list[0].shape={tuple(gt_c_list[0].shape)} "
              f"gt_c_list[1].shape={tuple(gt_c_list[1].shape)} "
              f"gt_decode_derived_c[1].shape={tuple(gt_c1_decode_derived.shape)}")
        raw_ids = gt_c_list[1].argmax(-1)[0, :16].tolist()
        derived_ids = gt_c1_decode_derived.argmax(-1)[0, :16].tolist()
        print(f"  [oracle] level1 raw encode ids[:16]     = {raw_ids}")
        print(f"  [oracle] level1 decode_derived ids[:16] = {derived_ids}  <- this is what cond_full actually conditions on (cross_track_source={cfg.cross_track_source!r})")

    K0 = cfg.Ks[0]
    window0 = model.decode_windows[0][0]
    K1_cum = K0 * cfg.Ks[1]
    window1 = model.decode_windows[0][1]

    all_bytes = prompt_bytes
    for step in range(n_new_bytes):
        L = all_bytes.shape[1]
        pad_len = (-L) % decode_K
        padded = (torch.cat([all_bytes, all_bytes.new_zeros(all_bytes.shape[0], pad_len)], dim=1)
                  if pad_len > 0 else all_bytes)
        Lp = padded.shape[1]

        # Self track: level 0's own encoder over the (free-running) generated-so-far bytes --
        # this part is NOT oracle, matching what generate_no_cache does for cond_full.
        c0_self, _, _, _, _ = model.encode_lms[0](padded, level=0, window=model.windows[0], compute_ntp=False)
        self_track = (_code_embed(model, 0, c0_self), K0, window0)

        # Coarse track: sliced from the REAL decode_derived_c[1] computed on ground truth above.
        n_units = Lp // K1_cum
        source_c = gt_c1_decode_derived[:, :n_units, :]
        coarse_track = (_code_embed(model, 0, source_c), K1_cum, window1)

        tracks = [self_track, coarse_track]
        if verbose and step < 3:
            print(f"  [oracle step {step}] L={L} Lp={Lp} n_units(coarse)={n_units} "
                  f"self_track_embed.shape={tuple(self_track[0].shape)} "
                  f"coarse_track_embed.shape={tuple(coarse_track[0].shape)}")

        c_out, _, _, h_out, query_last = model.decode_lms[0](
            padded, level=0, window=model.windows[0], compute_ntp=False,
            decode_tracks=tracks, extra_query=(pad_len == 0))
        query = query_last if pad_len == 0 and query_last is not None else h_out[:, L - 1, :]
        next_byte = _sample_next_byte(model.decode_lms[0].embed.weight, query)
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)

    if was_training:
        model.train()
    return all_bytes[0]


def _first_divergence(gen: bytes, gt: bytes) -> int:
    n = min(len(gen), len(gt))
    for i in range(n):
        if gen[i] != gt[i]:
            return i
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", default="checkpoints/qcute_v5_concat_1_verify/last.pt")
    p.add_argument("--offset", type=int, default=START)
    p.add_argument("--prompt_len", type=int, default=PROMPT_LEN)
    p.add_argument("--gen_len", type=int, default=GEN_LEN)
    p.add_argument("--data", type=str, default="datasets/enwik8_1M.gz")
    p.add_argument("--n_bytes", type=int, default=10000)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Config(**ckpt["cfg"])
    model = RefineLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_path} step={ckpt.get('step')} Ks={cfg.Ks} n_layers={cfg.n_layers} "
          f"cross_track_source={cfg.cross_track_source}")

    if model.n_levels != 2:
        print(f"n_levels={model.n_levels} != 2 -- this script's oracle chain only handles the "
              f"2-level case, skipping")
        return

    data = load_enwik8(Path(args.data), n_bytes=args.n_bytes)
    train_data, val_data = split_train_val(data, 0.1)

    for label, src in (("train", train_data), ("val", val_data)):
        offset = args.offset if label == "train" else 0
        window = src[offset:offset + args.prompt_len + args.gen_len]
        prompt = window[:args.prompt_len]
        gt = window[args.prompt_len:]
        full_gt = window  # prompt + ground-truth continuation

        out_free = generate_no_cache(model, prompt, args.gen_len, device)
        gen_free = bytes(out_free[args.prompt_len:].tolist())

        out_self = generate_self_only_cond(model, prompt, args.gen_len, device)
        gen_self = bytes(out_self[args.prompt_len:].tolist())

        print(f"\n--- {label} (offset={offset}) ---")
        out_oracle = generate_cond_full_oracle_coarse(model, prompt, full_gt, args.gen_len, device,
                                                        verbose=args.verbose)
        gen_oracle = bytes(out_oracle[args.prompt_len:].tolist())

        gt_bytes = bytes(gt.tolist())
        print(f"prompt:                {bytes(prompt.tolist())!r}")
        print(f"ground_truth:          {gt_bytes!r}")
        print(f"cond_full (free):      {gen_free!r}  first_diverge={_first_divergence(gen_free, gt_bytes)}")
        print(f"cond_self:             {gen_self!r}  first_diverge={_first_divergence(gen_self, gt_bytes)}")
        print(f"cond_full (oracle_l1): {gen_oracle!r}  first_diverge={_first_divergence(gen_oracle, gt_bytes)}")


if __name__ == "__main__":
    main()
