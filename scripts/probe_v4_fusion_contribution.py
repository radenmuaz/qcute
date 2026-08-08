"""Does qcute_refine_v4's fusion actually help a TRAINED checkpoint, and
does it need real content or just "something to attend to"?

Unlike qcute_refine_v2/v3's scripts/probe_decoder_kv_contribution.py
(which targets DecoderLevel — a module v4 doesn't have at all), this
probe is simpler and more direct: v4's fusion feeds byte_loss itself (not
a separate detached tok_loss), so we can just re-evaluate the SAME
trained checkpoint under three conditions and compare val_bpb/acc
directly — no gradient-norm proxies or attention-mass workarounds needed,
since fusion's effect on the metric that matters is now measurable head-on.

Four conditions, same weights throughout:
  1. normal    — fuse_encoder_levels=True, as trained (real PASS 2).
  2. no_fusion — fuse_encoder_levels=False (PASS 1 only, matching v2's
                 own byte_loss computation) — the full ablation.
  3. null_only — PASS 2 still runs, but fuse_kv is replaced with zeros
                 (same shape as the real level-above hidden state) before
                 quantization/projection — tests whether the model needs
                 REAL coarser-level content, or whether merely having a
                 KV tensor to cross-attend to (structurally) is most of
                 the benefit (a null/zero KV still lets Q attend to the
                 null_kv slot and to a degenerate "all zeros" set of real
                 rows, so this isolates content from structure).
  4. big_noise — PASS 2 runs with fuse_kv = REAL content + large-magnitude
                 i.i.d. Gaussian noise (10x h's own std) — simplest possible
                 corruption, no separate noise tensor/shape bookkeeping,
                 just drown out the real signal in-place. A STRONGER
                 control than null_only: zeros are a degenerate, structured
                 input a model could plausibly handle specially (e.g. via
                 LayerNorm/bias terms); large additive noise keeps real
                 variance/structure partially present but swamps the
                 actual coarser-level signal — isolates "needs a varying
                 signal to attend to" from "needs the SPECIFIC coarser-
                 level content," a distinction null_only alone can't
                 fully separate.

Usage:
    uv run python scripts/probe_v4_fusion_contribution.py \\
        --checkpoint checkpoints/qcute_refine_v4_k32_narrow/best.pt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qcute.qcute_refine_v4 import (
    Config, RefineLM, load_enwik8, split_train_val, sample_context,
)


def encode_with_ablation(model: RefineLM, byte_ids: torch.Tensor, n_active: int, mode: str):
    """Mirrors RefineLM._encode's own PASS 1 + PASS 2, with an injectable
    ablation on PASS 2's fuse_kv. mode: "normal" | "no_fusion" | "null_only"."""
    cfg = model.cfg
    seq_repr = byte_ids
    ntp_losses, ntp_accs, h_list, x_list = [], [], [], []

    for i in range(n_active):
        want_ntp = i == 0 or cfg.code_ntp_weight > 0
        c_i, ntp_loss, ntp_acc, h_i = model.encoders[i](seq_repr, compute_ntp=want_ntp)
        ntp_losses.append(ntp_loss); ntp_accs.append(ntp_acc)
        h_list.append(h_i); x_list.append(seq_repr)
        seq_repr = c_i

    if mode != "no_fusion" and cfg.fuse_encoder_levels:
        for i in range(n_active - 1):
            fuse_kv = h_list[i + 1].detach()
            if mode == "null_only":
                fuse_kv = torch.zeros_like(fuse_kv)
            elif mode == "big_noise":
                fuse_kv = fuse_kv + 10.0 * fuse_kv.std() * torch.randn_like(fuse_kv)
            c_i2, ntp_loss2, ntp_acc2, h_i2 = model.encoders[i](
                x_list[i], compute_ntp=True, fuse_kv=fuse_kv
            )
            ntp_losses[i] = ntp_loss2; ntp_accs[i] = ntp_acc2; h_list[i] = h_i2

    return ntp_losses, ntp_accs, h_list, x_list


@torch.no_grad()
def eval_mode(model: RefineLM, data: torch.Tensor, batch_size: int, n_batches: int, context_len: int, mode: str, device: str):
    model.eval()
    losses, accs = [], []
    for _ in range(n_batches):
        ctx = sample_context(data, batch_size, context_len, device)
        ntp_losses, ntp_accs, _, _ = encode_with_ablation(model, ctx, model.n_levels, mode)
        losses.append(ntp_losses[0].item())
        accs.append(ntp_accs[0].item())
    return sum(losses) / len(losses), sum(accs) / len(accs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--n_batches", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = Config(**ckpt["cfg"])
    model = RefineLM(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.checkpoint} (step {ckpt.get('step')}), Ks={cfg.Ks} attn_window={cfg.attn_window}")

    import math
    data = load_enwik8(args.data)
    train_data, val_data = split_train_val(data, args.val_frac)

    results = {}
    for mode in ["normal", "null_only", "big_noise", "no_fusion"]:
        loss, acc = eval_mode(model, val_data, args.batch_size, args.n_batches, cfg.context_len, mode, args.device)
        results[mode] = (loss, acc, loss / math.log(2))
        print(f"{mode:12s}  val_byte_loss={loss:.4f}  val_byte_acc={acc:.4f}  val_bpb={loss/math.log(2):.4f}")

    print()
    print(f"delta_bpb (normal - null_only)    = {results['normal'][2] - results['null_only'][2]:+.4f}  (content vs. structure-only)")
    print(f"delta_bpb (normal - big_noise)    = {results['normal'][2] - results['big_noise'][2]:+.4f}  (content vs. drowned-out-content)")
    print(f"delta_bpb (normal - no_fusion)    = {results['normal'][2] - results['no_fusion'][2]:+.4f}  (fusion's total contribution)")
    print(f"delta_bpb (null_only - no_fusion) = {results['null_only'][2] - results['no_fusion'][2]:+.4f}  (structure-only vs. nothing)")
    print(f"delta_bpb (big_noise - no_fusion) = {results['big_noise'][2] - results['no_fusion'][2]:+.4f}  (noisy-structure vs. nothing)")


if __name__ == "__main__":
    main()
