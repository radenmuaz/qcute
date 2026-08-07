"""scripts/probe_decoder_kv_contribution.py — diagnoses how much
DecoderLevel's cross-attention KV (EncoderLevel[level+1]'s own hidden
states — the coarser code) actually contributes to the final prediction,
vs. being effectively ignored (the model relying only on its own causal
Q-side history). Run AFTER training finishes, against a saved
checkpoint, probing several REAL samples from both train and val data
(not synthetic — session ask: "probe with several train samples" + "and
val").

Session motivation: qcute_refine_v2_byte4_code256_simple's own live
metrics showed level0_ntp_acc (EncoderLevel alone, no cross-attention)
and pair0_tok_acc (DecoderLevel, cross-attends to EncoderLevel[1]'s
code) nearly identical mid-training (49.1% vs 49.4%) — this script gives
a rigorous, per-checkpoint answer instead of eyeballing two aggregate
accuracy numbers logged from different heads.

Three independent signals, since no single one is fully trustworthy
alone:
  1. GRADIENT norm: d(loss)/d(h_curr) vs d(loss)/d(h_prev) at the
     DecoderLevel's own input boundary. A large ratio in favor of h_prev
     suggests the model barely uses h_curr — but gradient magnitude
     alone can understate a real (nonlinear) causal effect, so it's
     reported alongside, not instead of, signal 2.
  2. ABLATION: rerun with ALL real code-block KV positions masked out —
     forcing every query to attend ONLY to the trained null-KV slot,
     i.e. simulating "no coarser code available at all" using the
     model's own learned null fallback (the same one it already uses for
     block 0's own left-edge case in normal training). The loss/accuracy
     delta vs. the real forward pass is the most directly interpretable,
     CAUSAL measure of what the cross-attention KV is actually buying.
  3. ATTENTION WEIGHT mass on the null slot vs. real code slots — direct
     structural evidence of whether the cross-attention softmax is even
     looking at the real KV content. Requires manually recomputing the
     cross-attention scores (CrossBlock's own forward uses
     F.scaled_dot_product_attention internally, which doesn't expose
     attention weights), using the exact same trained q_proj/kv_proj
     weights and causal-block-visibility mask as the real forward pass,
     so the numbers reflect what the model actually computed, not an
     approximation.

    uv run python scripts/probe_decoder_kv_contribution.py \\
        --checkpoint checkpoints/qcute_refine_v2_byte4_code256_simple/best.pt \\
        --n_samples 8
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from qcute.qcute_refine_v2 import Config, RefineLM, load_enwik8, sample_context, split_train_val


@torch.no_grad()
def _attn_weights(decoder, h_prev: torch.Tensor, h_curr: torch.Tensor) -> torch.Tensor:
    """Manually recomputes DecoderLevel's cross-attention softmax weights
    using its own trained sub-modules and mask — CrossBlock.forward()
    itself never exposes them (F.scaled_dot_product_attention doesn't
    return weights). Returns [B, H, L, 1+n_blocks] (index 0 = null slot,
    1: = real code blocks)."""
    cfg = decoder.cfg
    K = decoder.K
    B, L, _ = h_prev.shape
    n_blocks = h_curr.size(1)
    D = cfg.tok_d_model
    cb = decoder.cross_block

    q_in = decoder.q_proj(h_prev)
    kv_in = decoder.kv_proj(h_curr)
    null = decoder.null_kv.expand(B, 1, D)
    kv_in = torch.cat([null, kv_in], dim=1)

    q_n, kv_n = cb.ln_q(q_in), cb.ln_kv(kv_in)
    H, hd = cb.n_heads, cb.head_dim
    qh = cb.q_proj(q_n).reshape(B, L, H, hd).transpose(1, 2)
    kvp = cb.kv_proj(kv_n).reshape(B, 1 + n_blocks, 2, H, hd).permute(2, 0, 3, 1, 4)
    kh = kvp[0]

    t_idx = torch.arange(L, device=h_prev.device).unsqueeze(1)
    b_idx = torch.arange(n_blocks, device=h_prev.device).unsqueeze(0)
    visible = b_idx < ((t_idx + 1) // K)
    null_col = torch.ones(L, 1, dtype=torch.bool, device=h_prev.device)
    visible = torch.cat([null_col, visible], dim=1)   # [L, 1+n_blocks]

    scores = torch.einsum("bhqd,bhkd->bhqk", qh, kh) / (hd ** 0.5)
    scores = scores.masked_fill(~visible.unsqueeze(0).unsqueeze(0), float("-inf"))
    return F.softmax(scores, dim=-1)


def _decode_ablated_no_kv(decoder, h_prev: torch.Tensor, h_curr: torch.Tensor, seq_repr: torch.Tensor):
    """Same computation as DecoderLevel.forward, except EVERY real code
    block is masked out — only the null slot is ever visible, regardless
    of position. Duplicated here (not a flag on DecoderLevel itself) to
    keep this ablation a probe-only concept, not a training-time option."""
    cfg = decoder.cfg
    B, L, _ = h_prev.shape
    n_blocks = h_curr.size(1)
    D = cfg.tok_d_model

    q = decoder.q_proj(h_prev)
    kv = decoder.kv_proj(h_curr)
    null = decoder.null_kv.expand(B, 1, D)
    kv = torch.cat([null, kv], dim=1)

    disallow = torch.ones(L, 1 + n_blocks, dtype=torch.bool, device=h_prev.device)
    disallow[:, 0] = False   # only the null slot (index 0) is ever visible

    h_dec = decoder.cross_block(q, kv, attn_mask=disallow)
    h_flat = h_dec[:, :-1, :].reshape(-1, D)
    if decoder.use_byte_softmax:
        target = seq_repr[:, 1:].reshape(-1)
        logits = decoder.head(h_flat)
        loss = F.cross_entropy(logits, target)
        acc = (logits.argmax(-1) == target).float().mean()
    else:
        from qcute.qcute_refine_v2 import byte_to_bits, chain_bce_loss
        true_seq = byte_to_bits(seq_repr) if decoder.level == 0 else seq_repr
        true_flat = true_seq[:, 1:, :].reshape(-1, decoder.in_dq)
        raw = decoder.head(h_flat, true_flat) if cfg.tok_head_mode == "chain" else decoder.head(h_flat)
        loss = chain_bce_loss(raw, true_flat)
        acc = ((raw > 0) == (true_flat > 0)).float().mean()
    return loss, acc


def probe_pair(model: RefineLM, pair_idx: int, ctx: torch.Tensor) -> dict:
    decoder = model.decoders[pair_idx]

    seq_repr = ctx
    h_list, x_list = [], []
    with torch.no_grad():
        for i in range(pair_idx + 2):
            c_i, _, _, h_i = model.encoders[i](seq_repr, compute_ntp=False)
            h_list.append(h_i)
            x_list.append(seq_repr)
            seq_repr = c_i

    h_prev = h_list[pair_idx].detach().clone().requires_grad_(True)
    h_curr = h_list[pair_idx + 1].detach().clone().requires_grad_(True)
    target_seq = x_list[pair_idx]

    loss, acc = decoder(h_prev, h_curr, target_seq)
    grad_prev, grad_curr = torch.autograd.grad(loss, [h_prev, h_curr])
    grad_prev_norm = grad_prev.norm().item()
    grad_curr_norm = grad_curr.norm().item()

    with torch.no_grad():
        loss_ablated, acc_ablated = _decode_ablated_no_kv(decoder, h_prev, h_curr, target_seq)
        weights = _attn_weights(decoder, h_prev, h_curr)
        null_mass = weights[..., 0].mean().item()

    return {
        "pair": pair_idx,
        "loss": loss.item(), "acc": acc.item(),
        "loss_ablated_no_kv": loss_ablated.item(), "acc_ablated_no_kv": acc_ablated.item(),
        "delta_loss_from_kv": loss.item() - loss_ablated.item(),   # positive = KV genuinely helps
        "delta_acc_from_kv": acc.item() - acc_ablated.item(),
        "grad_norm_h_prev": grad_prev_norm, "grad_norm_h_curr": grad_curr_norm,
        "grad_ratio_curr_over_prev": grad_curr_norm / max(grad_prev_norm, 1e-12),
        "null_slot_attn_mass": null_mass,   # close to 1.0 = attention barely looks at real code blocks
    }


def run_split(model: RefineLM, data: torch.Tensor, split_name: str, n_samples: int, batch_size: int, device: str, log) -> None:
    accum: dict[tuple[int, str], list[float]] = {}
    for _ in range(n_samples):
        ctx = sample_context(data, batch_size, model.cfg.context_len, device)
        for pair_idx in range(model.n_levels - 1):
            result = probe_pair(model, pair_idx, ctx)
            for k, v in result.items():
                if k == "pair":
                    continue
                accum.setdefault((pair_idx, k), []).append(v)

    log(f"\n=== {split_name} ({n_samples} batches x {batch_size}) ===")
    for pair_idx in range(model.n_levels - 1):
        log(f"-- pair {pair_idx} (Q=encoders[{pair_idx}], KV=encoders[{pair_idx + 1}]) --")
        for metric in ["loss", "loss_ablated_no_kv", "delta_loss_from_kv", "acc", "acc_ablated_no_kv",
                        "delta_acc_from_kv", "grad_norm_h_prev", "grad_norm_h_curr",
                        "grad_ratio_curr_over_prev", "null_slot_attn_mass"]:
            vals = accum[(pair_idx, metric)]
            mean = sum(vals) / len(vals)
            log(f"  {metric:28s} mean={mean:.4f}  min={min(vals):.4f}  max={max(vals):.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--n_samples", type=int, default=8, help="number of batches to probe, per split")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    args = p.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = Config(**ckpt["cfg"])
    model = RefineLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint} (step {ckpt.get('step')}), n_levels={model.n_levels}, device={device}")

    data = load_enwik8(args.data)
    train_data, val_data = split_train_val(data, args.val_frac)

    def log(msg):
        print(msg)

    run_split(model, train_data, "train", args.n_samples, args.batch_size, device, log)
    run_split(model, val_data, "val", args.n_samples, args.batch_size, device, log)


if __name__ == "__main__":
    main()
