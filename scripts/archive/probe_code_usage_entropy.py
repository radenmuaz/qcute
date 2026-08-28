"""Diagnose codebook/index collapse directly: for each checkpoint, run the encode pass over a
large chunk of the real validation set, histogram which code indices (0..vocab-1) get used by
level0's own code_0 output, and report entropy (bits) + effective vocab size (2**entropy) +
top-5 index mass. A healthy, non-collapsed code should have entropy close to log2(vocab); a
collapsed one will show entropy near 0 and >90% mass on a handful of indices -- exactly what the
qualitative single-prompt samples were hinting at (repeating {213}/{170}/{58}-style codes) but
this quantifies it precisely over ~thousands of code tokens instead of eyeballing 16 at a time.

Also reports level1's OWN code_1 entropy where applicable (n_levels>1), to separate "is code_0
itself collapsed" from "is level1's separate quantizer ALSO collapsed" -- two distinct mechanisms
sharing the same collapse symptom.

CPU-only, no MPS contention with any training job.

    uv run python scripts/probe_code_usage_entropy.py
"""
import gzip
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qcute.qcute_refine_v4_4 import Config, RefineLM

device = "cpu"
REPO = Path(__file__).resolve().parent.parent

CHECKPOINTS = [
    ("1level (Ks=(4,))", REPO / "checkpoints/qcute_refine_v4_4_bpelike_1level_k4_retry/best.pt"),
    ("2level cond_full+cond_self (old)", REPO / "checkpoints/qcute_refine_v4_4_bpelike_k4_1/best.pt"),
    ("2level +decode_self_only_aux", REPO / "checkpoints/qcute_refine_v4_4_bpelike_k4_1_selfonly_aux/best.pt"),
    ("2level self-only-ONLY (cross disabled)", REPO / "checkpoints/qcute_refine_v4_4_bpelike_k4_1_selfonly_only/best.pt"),
    ("2level selfcond_detach (rerun, real detach)", REPO / "checkpoints/qcute_refine_v4_4_selfcond_detach_k4_rerun/best.pt"),
    ("past-success v4.4 l1_k1", REPO / "checkpoints/qcute_refine_v4_4_l1_k1/best.pt"),
    ("past-success v4.4 l2_k1", REPO / "checkpoints/qcute_refine_v4_4_l2_k1/best.pt"),
    ("k1_k1_w32 (Ks=(1,1), no gumbel)", REPO / "checkpoints/qcute_refine_v4_4_k1_k1_w32/best.pt"),
    ("k1_k1_w32_gumbel (Ks=(1,1), gumbel=True)", REPO / "checkpoints/qcute_refine_v4_4_k1_k1_w32_gumbel/best.pt"),
    ("bpelike_k4_1_gumbelfix (Ks=(4,1), gumbel=True)", REPO / "checkpoints/qcute_refine_v4_4_bpelike_k4_1_gumbelfix/best.pt"),
    ("bpelike_1level_k4_gumbel (Ks=(4,), gumbel=True)", REPO / "checkpoints/qcute_refine_v4_4_bpelike_1level_k4_gumbel/best.pt"),
    ("bpelike_k4_1_crosstrack_decode (cross_track_source=decode)", REPO / "checkpoints/qcute_refine_v4_4_bpelike_k4_1_crosstrack_decode/best.pt"),
]


def entropy_bits(counts: torch.Tensor) -> float:
    p = counts.float() / counts.sum().clamp(min=1)
    p = p[p > 0]
    return float(-(p * p.log2()).sum())


@torch.no_grad()
def code_usage(model: RefineLM, bytes_tensor: torch.Tensor, level: int, vocab: int) -> torch.Tensor:
    model.eval()
    seq_repr = bytes_tensor
    for i in range(level + 1):
        c_i, _, _, _ = model.encoders[i](seq_repr, compute_ntp=False)
        seq_repr = c_i
    ids = seq_repr.argmax(-1).reshape(-1)
    return torch.bincount(ids, minlength=vocab)


def main():
    with gzip.open(REPO / "datasets/enwik8_1M.gz", "rb") as f:
        data = f.read()
    n_val = int(len(data) * 0.1)
    val_data = data[-n_val:]

    for name, path in CHECKPOINTS:
        if not path.exists():
            print(f"{name}: SKIPPED ({path.name} not found)")
            continue
        ckpt = torch.load(path, map_location=device)
        cfg = Config(**ckpt["cfg"])
        model = RefineLM(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        L = cfg.context_len
        n_chunks = min(20, len(val_data) // L)
        chunks = torch.stack([
            torch.tensor(list(val_data[i * L:(i + 1) * L]), dtype=torch.long) for i in range(n_chunks)
        ])

        print(f"\n{'='*20} {name} (step={ckpt['step']}, Ks={cfg.Ks}, vocab={cfg.vocab}) {'='*20}")
        max_entropy = math.log2(cfg.vocab)
        for level in range(cfg.__dict__.get("n_levels", len(cfg.Ks))) if False else range(len(cfg.Ks)):
            counts = code_usage(model, chunks, level, cfg.vocab)
            ent = entropy_bits(counts)
            top5 = counts.topk(min(5, cfg.vocab))
            top5_mass = top5.values.sum().item() / counts.sum().item()
            n_active = int((counts > 0).sum())
            print(f"  level{level} code_{level}: entropy={ent:.2f}/{max_entropy:.2f} bits "
                  f"(effective_vocab={2**ent:.1f}/{cfg.vocab}), active_indices={n_active}/{cfg.vocab}, "
                  f"top5_mass={top5_mass*100:.1f}%, top5_idx={top5.indices.tolist()}")


if __name__ == "__main__":
    main()
