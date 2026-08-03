"""Diagnose a trained qcute.qcutelm (BSQ, tightly-coupled) checkpoint beyond
the aggregate recon_acc/latent_acc numbers already logged during training:

1. Per-position (byte 0..K-1) reconstruction accuracy, both for the main
   z_pred path and (if aux_recon was on) the aux z_hat path. Checks whether
   any one byte position within the chunk is systematically worse than the
   others — motivated by an earlier (reverted) hypothesis that byte 0 might
   be under-served; see docs/status.md's "encoder reverted to plain MLP"
   entry for why that hypothesis was dropped in favor of measuring first.

2. Per-loss-term gradient norm (rec_loss / pred_loss / aux_rec_loss) over
   model.parameters() on a single batch — checks whether one loss term is
   dominating the shared gradient into the encoder/decoder, which unweighted
   loss = rec_loss + pred_loss (+ aux_rec_loss) doesn't prevent by design.

    uv run python scripts/diagnose_qcutelm.py --checkpoint_path checkpoints/qcutelm_bsq_k4_lfq_aux/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from qcute.qcutelm import (
    Config,
    QCuteLM,
    batch_iter,
    bsq_quantize,
    load_enwik8,
    maskgit_mask,
    split_train_val,
)


@torch.no_grad()
def per_position_breakdown(model: QCuteLM, data_iter, n_batches: int) -> dict:
    """Per-position accuracy is now measured only over positions that
    happened to be masked that batch (MaskGIT decoder never sees/predicts
    unmasked positions) — averaged over enough batches, every position gets
    covered many times regardless of the random per-batch mask draw."""
    model.eval()
    cfg = model.cfg
    K = cfg.K
    main_correct = torch.zeros(K)
    aux_correct = torch.zeros(K)
    main_total = torch.zeros(K)
    aux_total = torch.zeros(K)
    for _ in range(n_batches):
        batch = next(data_iter)  # [B, T, K]
        B, T, _ = batch.shape
        z_hat, targets = model.encoder(batch.reshape(B * T, K))
        z_hat, targets = z_hat.reshape(B, T, -1), targets.reshape(B, T, -1)

        v_pred = model.lm(z_hat)[:, :-1]
        z_pred, _ = bsq_quantize(v_pred, cfg.dq, cfg.lfq)
        true_next_bytes = batch[:, 1:].reshape(-1, K)
        x_masked, mask = maskgit_mask(true_next_bytes, model.decoder.mask_id)
        rec_logits = model.decoder(z_pred.reshape(-1, cfg.dq), x_masked)
        correct = (rec_logits.argmax(-1) == true_next_bytes) & mask
        main_correct += correct.float().sum(dim=0).cpu()
        main_total += mask.float().sum(dim=0).cpu()

        if cfg.aux_recon:
            flat_bytes = batch.reshape(-1, K)
            aux_x_masked, aux_mask = maskgit_mask(flat_bytes, model.decoder.mask_id)
            aux_logits = model.decoder(z_hat.reshape(-1, cfg.dq), aux_x_masked)
            aux_correct_this = (aux_logits.argmax(-1) == flat_bytes) & aux_mask
            aux_correct += aux_correct_this.float().sum(dim=0).cpu()
            aux_total += aux_mask.float().sum(dim=0).cpu()
    model.train()
    result = {"main_recon_acc_per_position": (main_correct / main_total).tolist()}
    if cfg.aux_recon:
        result["aux_recon_acc_per_position"] = (aux_correct / aux_total).tolist()
    return result


def grad_norms_per_loss(model: QCuteLM, batch: torch.Tensor) -> dict[str, float]:
    """Recomputes the tightly-coupled forward pass's three loss terms as
    separate tensors (the trained forward() only exposes their sum), then
    backprops each individually to measure ||grad|| over all parameters."""
    model.train()
    cfg = model.cfg
    B, T, K = batch.shape
    z_hat, targets = model.encoder(batch.reshape(B * T, K))
    z_hat, targets = z_hat.reshape(B, T, -1), targets.reshape(B, T, -1)

    v_pred = model.lm(z_hat)[:, :-1]
    z_pred, pred_bits = bsq_quantize(v_pred, cfg.dq, cfg.lfq)
    pred_targets = targets[:, 1:]
    pred_loss = F.binary_cross_entropy_with_logits(v_pred, pred_targets)

    true_next_bytes = batch[:, 1:].reshape(-1, K)
    x_masked, mask = maskgit_mask(true_next_bytes, model.decoder.mask_id)
    rec_logits = model.decoder(z_pred.reshape(-1, cfg.dq), x_masked)
    rec_loss = F.cross_entropy(rec_logits[mask], true_next_bytes[mask])

    losses = {"rec_loss": rec_loss, "pred_loss": pred_loss}
    if cfg.aux_recon:
        flat_bytes = batch.reshape(-1, K)
        aux_x_masked, aux_mask = maskgit_mask(flat_bytes, model.decoder.mask_id)
        aux_logits = model.decoder(z_hat.reshape(B * T, cfg.dq), aux_x_masked)
        aux_rec_loss = F.cross_entropy(aux_logits[aux_mask], flat_bytes[aux_mask])
        losses["aux_rec_loss"] = aux_rec_loss

    norms = {}
    names = list(losses)
    for i, name in enumerate(names):
        model.zero_grad(set_to_none=True)
        losses[name].backward(retain_graph=(i < len(names) - 1))
        total = sum(p.grad.detach().pow(2).sum().item() for p in model.parameters() if p.grad is not None)
        norms[name] = total ** 0.5
    model.zero_grad(set_to_none=True)
    return norms


def main():
    p = argparse.ArgumentParser(description="Diagnose a qcute.qcutelm BSQ checkpoint")
    p.add_argument("--checkpoint_path", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--n_bytes", type=int, default=2_000_000)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_chunks", type=int, default=32)
    p.add_argument("--eval_batches", type=int, default=20)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    cfg = Config(**ckpt["cfg"])
    if cfg.bottleneck != "bsq":
        raise SystemExit(f"this diagnostic only supports bottleneck=bsq (tightly-coupled path); got {cfg.bottleneck}")
    model = QCuteLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.checkpoint_path}  bottleneck={cfg.bottleneck} lfq={cfg.lfq} aux_recon={cfg.aux_recon} "
          f"dq={cfg.dq} K={cfg.K}  step={ckpt.get('step')}")

    data = load_enwik8(args.data, args.n_bytes)
    _, val_data = split_train_val(data, args.val_frac)
    val_iter = batch_iter(val_data, args.batch_size, args.seq_chunks, cfg.K, device)

    breakdown = per_position_breakdown(model, val_iter, args.eval_batches)
    print("\nper-position recon_acc (byte 0 = first byte in chunk):")
    for i, acc in enumerate(breakdown["main_recon_acc_per_position"]):
        print(f"  position {i} (main, z_pred): {acc*100:.2f}%")
    if "aux_recon_acc_per_position" in breakdown:
        for i, acc in enumerate(breakdown["aux_recon_acc_per_position"]):
            print(f"  position {i} (aux,  z_hat):  {acc*100:.2f}%")

    batch = next(val_iter)
    norms = grad_norms_per_loss(model, batch)
    print("\nper-loss-term gradient norm (||grad|| over all model.parameters(), single batch):")
    for name, norm in norms.items():
        print(f"  {name}: {norm:.4f}")


if __name__ == "__main__":
    main()
