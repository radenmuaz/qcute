"""Ablation: decode_code_ste=False (detach, selfcond_detach_k4_rerun) vs True (straight-through,
selfcond_ste_k4) -- otherwise identical config (self-conditioning-only, level1 as pure NTP
drafter, level1's own decode disabled). Compares val metrics from each run's own log (loss/acc/bpb
for level0 decode and level1 encode -- the drafter's own accuracy at predicting code_0) plus code
usage entropy for both code_0 and level1's implicit prediction target, via the same methodology as
scripts/probe_code_usage_entropy.py.

    uv run python scripts/compare_ste_vs_detach.py
"""
import gzip
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qcute.qcute_refine_v4_4 import Config, RefineLM

device = "cpu"
REPO = Path(__file__).resolve().parent.parent

RUNS = [
    ("detach (decode_code_ste=False)", "qcute_refine_v4_4_selfcond_detach_k4_rerun"),
    ("STE (decode_code_ste=True)", "qcute_refine_v4_4_selfcond_ste_k4"),
]


def last_val_line(run_name: str) -> dict:
    log_path = REPO / "logs" / run_name / "run.log"
    text = log_path.read_text().strip().splitlines()
    for line in reversed(text):
        if "best_val_bpb=" in line:
            return {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", line) if k not in ("step",)}
    return {}


def entropy_bits(counts: torch.Tensor) -> float:
    p = counts.float() / counts.sum().clamp(min=1)
    p = p[p > 0]
    return float(-(p * p.log2()).sum())


@torch.no_grad()
def code_entropy(run_name: str, val_data: bytes) -> dict:
    ckpt_path = REPO / "checkpoints" / run_name / "best.pt"
    if not ckpt_path.exists():
        return {}
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Config(**ckpt["cfg"])
    model = RefineLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    L = cfg.context_len
    n_chunks = min(20, len(val_data) // L)
    chunks = torch.stack([
        torch.tensor(list(val_data[i * L:(i + 1) * L]), dtype=torch.long) for i in range(n_chunks)
    ])
    seq_repr = chunks
    out = {}
    for i in range(len(cfg.Ks)):
        c_i, _, _, _ = model.encoders[i](seq_repr, compute_ntp=False)
        ids = c_i.argmax(-1).reshape(-1)
        counts = torch.bincount(ids, minlength=cfg.vocab)
        out[f"code_{i}_entropy_bits"] = entropy_bits(counts)
        out[f"code_{i}_active"] = int((counts > 0).sum())
        seq_repr = c_i
    out["step"] = ckpt["step"]
    return out


def main():
    with gzip.open(REPO / "datasets/enwik8_1M.gz", "rb") as f:
        data = f.read()
    n_val = int(len(data) * 0.1)
    val_data = data[-n_val:]

    for label, run_name in RUNS:
        print(f"\n{'='*20} {label} ({run_name}) {'='*20}")
        val_metrics = last_val_line(run_name)
        if not val_metrics:
            print("  no val metrics found (run not finished?)")
            continue
        for k in ("val_bpb", "val_level0_ntp_acc_decode", "val_level0_ntp_loss_decode",
                   "val_level1_ntp_acc_encode", "val_level1_ntp_loss_encode", "best_val_bpb"):
            if k in val_metrics:
                print(f"  {k}: {val_metrics[k]:.4f}")
        ent = code_entropy(run_name, val_data)
        if ent:
            print(f"  step={ent['step']}  code_0: {ent['code_0_entropy_bits']:.2f} bits "
                  f"({ent['code_0_active']}/256 active)  "
                  f"code_1: {ent['code_1_entropy_bits']:.2f} bits ({ent['code_1_active']}/256 active)")


if __name__ == "__main__":
    main()
