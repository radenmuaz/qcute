"""
Ablation eval for a qcute_v5_concat checkpoint: val bpb under (a) unconditioned byte LM (encode_lms[0]
alone, no cross-attention to any code), (b) full normal decode conditioning, (c) decode conditioning
with level>=1 codes replaced by random ids, self (level-0) codes replaced by random ids, or both --
to check how much the learned codes actually matter vs. random ones.

uv run python scripts/ablate_v5_concat.py --checkpoint checkpoints/qcute_v5_concat_1/best.pt \
    --config configs/qcute_v5_concat_1.py
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qcute.qcute_v5_concat import Config, RefineLM, load_enwik8, split_train_val, sample_context


def random_onehot_like(source_c: torch.Tensor) -> torch.Tensor:
    V = source_c.shape[-1]
    ids = torch.randint(0, V, source_c.shape[:-1], device=source_c.device)
    return torch.nn.functional.one_hot(ids, num_classes=V).to(source_c.dtype)


@torch.no_grad()
def encode_pass(model: RefineLM, byte_ids: torch.Tensor):
    cfg = model.cfg
    seq_repr = byte_ids
    h_list, c_list, x_list = [], [], []
    for i in range(model.n_levels):
        c_i, _, _, h_i, _, _ = model.encode_lms[i](seq_repr, level=i, window=model.windows[i], compute_ntp=False)
        h_list.append(h_i)
        c_list.append(c_i)
        x_list.append(seq_repr)
        seq_repr = c_i
    return h_list, c_list, x_list


@torch.no_grad()
def decode_upper_levels(model: RefineLM, x_list, c_list):
    """Run decode normally for every level above 0, so decode_derived_c[1] etc. are the real
    (decode-refined) codes level-0's cross track would otherwise consume."""
    cfg = model.cfg
    decode_derived_c: dict[int, torch.Tensor] = {}
    for i in reversed(range(1, model.n_levels)):
        L_i = x_list[i].shape[1]
        tracks = []
        cum_K = 1
        for j in range(i, model.n_levels):
            cum_K *= cfg.Ks[j]
            window = model.decode_windows[i][j - i]
            if window == 0:
                continue
            if L_i // cum_K < 1:
                break
            source_c = decode_derived_c[j] if (j > i and j in decode_derived_c) else c_list[j]
            dec_lm = model.decode_lms[i]
            code_embeds = dec_lm.quant.embed_for_decode(dec_lm, source_c)
            tracks.append((code_embeds, cum_K, window))
        if not tracks:
            continue
        if len(tracks) == 1 and (L_i // tracks[0][1]) < 2:
            continue
        c_i2, _, _, _, _, _ = model.decode_lms[i](
            x_list[i], level=i, window=model.windows[i], compute_ntp=False,
            decode_tracks=tracks, extra_query=False)
        decode_derived_c[i] = c_i2
    return decode_derived_c


@torch.no_grad()
def level0_byte_loss(model: RefineLM, x_list, c_list, decode_derived_c,
                      perturb_self: bool, perturb_cross: bool, override_cross: dict | None = None) -> torch.Tensor:
    cfg = model.cfg
    L_0 = x_list[0].shape[1]
    tracks = []
    cum_K = 1
    for j in range(model.n_levels):
        cum_K *= cfg.Ks[j]
        window = model.decode_windows[0][j]
        if window == 0:
            continue
        if L_0 // cum_K < 1:
            break
        if j > 0 and override_cross is not None and j in override_cross:
            source_c = override_cross[j]
        else:
            source_c = decode_derived_c[j] if (j > 0 and j in decode_derived_c) else c_list[j]
        if (j == 0 and perturb_self) or (j > 0 and perturb_cross):
            source_c = random_onehot_like(source_c)
        dec_lm = model.decode_lms[0]
        code_embeds = dec_lm.quant.embed_for_decode(dec_lm, source_c)
        tracks.append((code_embeds, cum_K, window))
    _, ntp_loss, _, _, _, _ = model.decode_lms[0](
        x_list[0], level=0, window=model.windows[0], compute_ntp=True,
        decode_tracks=tracks, extra_query=False)
    return ntp_loss


@torch.no_grad()
def ar_level1_codes(model: RefineLM, c0_full: torch.Tensor) -> torch.Tensor:
    """Genuine autoregressive rollout of level-1 codes: block 0's code is the real (ground-truth)
    encode output (no context exists to predict it from), every later code is enc1's own argmax
    prediction fed back in -- unlike decode_upper_levels' teacher-forced parallel pass, which
    conditions each position on the GROUND-TRUTH previous codes, not the model's own past guesses."""
    enc1 = model.encode_lms[1]
    B, n_blocks, V = c0_full.shape
    codes = c0_full[:, :1, :]
    for _ in range(n_blocks - 1):
        _, _, _, h1, _, _ = enc1(codes, level=1, window=model.windows[1], compute_ntp=False)
        next_code = enc1.quant.sample_next(enc1, h1[:, -1, :], model.cfg.vocab)
        codes = torch.cat([codes, next_code.unsqueeze(1)], dim=1)
    return codes


@torch.no_grad()
def uncond_byte_loss(model: RefineLM, byte_ids: torch.Tensor) -> torch.Tensor:
    _, ntp_loss, _, _, _, _ = model.encode_lms[0](byte_ids, level=0, window=model.windows[0], compute_ntp=True)
    return ntp_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/qcute_v5_concat_1/best.pt"))
    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_batches", type=int, default=20)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = Config(**ckpt["cfg"])
    model = RefineLM(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint} (step={ckpt['step']}, Ks={cfg.Ks})")

    data = load_enwik8(args.data)
    _, val_data = split_train_val(data, args.val_frac)

    modes = ["uncond", "full_cond", "perturb_cross", "perturb_self", "perturb_both", "ar_level1_cond"]
    accum = {m: [] for m in modes}

    for bi in range(args.n_batches):
        byte_ids = sample_context(val_data, args.batch_size, model.cfg.context_len, args.device)
        h_list, c_list, x_list = encode_pass(model, byte_ids)
        decode_derived_c = decode_upper_levels(model, x_list, c_list)

        accum["uncond"].append(uncond_byte_loss(model, byte_ids).item())
        accum["full_cond"].append(
            level0_byte_loss(model, x_list, c_list, decode_derived_c, perturb_self=False, perturb_cross=False).item())
        accum["perturb_cross"].append(
            level0_byte_loss(model, x_list, c_list, decode_derived_c, perturb_self=False, perturb_cross=True).item())
        accum["perturb_self"].append(
            level0_byte_loss(model, x_list, c_list, decode_derived_c, perturb_self=True, perturb_cross=False).item())
        accum["perturb_both"].append(
            level0_byte_loss(model, x_list, c_list, decode_derived_c, perturb_self=True, perturb_cross=True).item())

        ar_codes = ar_level1_codes(model, c_list[1])
        accum["ar_level1_cond"].append(
            level0_byte_loss(model, x_list, c_list, decode_derived_c, perturb_self=False, perturb_cross=False,
                              override_cross={1: ar_codes}).item())
        print(f"  batch {bi + 1}/{args.n_batches} done", flush=True)

    print(f"\n{'mode':<16} {'loss':>8} {'bpb':>8}")
    for m in modes:
        loss = sum(accum[m]) / len(accum[m])
        bpb = loss / math.log(2)
        print(f"{m:<16} {loss:>8.4f} {bpb:>8.4f}")


if __name__ == "__main__":
    main()
