"""Load several v4.4 checkpoints and compare generation quality side by side (same real
validation-set prompt for each), plus print each checkpoint's best_val_bpb for reference.
CPU-only (eval-only, small models -- no need to touch MPS / contend with any training job).

    uv run python scripts/compare_v4_4_checkpoints.py
"""
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qcute.qcute_refine_v4_4 import Config, RefineLM, qualitative_generate

device = "cpu"
REPO = Path(__file__).resolve().parent.parent

CHECKPOINTS = [
    ("1level (Ks=(4,))", REPO / "checkpoints/qcute_refine_v4_4_bpelike_1level_k4_retry/best.pt"),
    ("2level cond_full+cond_self (Ks=(4,1), old)", REPO / "checkpoints/qcute_refine_v4_4_bpelike_k4_1/best.pt"),
    ("2level +decode_self_only_aux", REPO / "checkpoints/qcute_refine_v4_4_bpelike_k4_1_selfonly_aux/best.pt"),
    ("2level self-only-ONLY (cross disabled)", REPO / "checkpoints/qcute_refine_v4_4_bpelike_k4_1_selfonly_only/best.pt"),
    ("past-success v4.4 l1_k1", REPO / "checkpoints/qcute_refine_v4_4_l1_k1/best.pt"),
    ("past-success v4.4 l2_k1", REPO / "checkpoints/qcute_refine_v4_4_l2_k1/best.pt"),
]


def load(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Config(**ckpt["cfg"])
    model = RefineLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt["step"]


def main():
    with gzip.open(REPO / "datasets/enwik8_1M.gz", "rb") as f:
        data = f.read()
    n_val = int(len(data) * 0.1)
    val_data = data[-n_val:]
    offset = 12345
    total_len = 128
    window = torch.tensor(list(val_data[offset:offset + total_len]), dtype=torch.long)
    prompt, gt = window[:64], window[64:128]

    for name, path in CHECKPOINTS:
        if not path.exists():
            print(f"\n{'='*20} {name} {'='*20}\n  SKIPPED: {path} not found")
            continue
        print(f"\n{'='*20} {name} {'='*20}")
        model, cfg, step = load(path)
        print(f"  step={step} Ks={cfg.Ks} context_len={cfg.context_len}")
        with torch.no_grad():
            qualitative_generate(model, prompt, 64, gt, device, log=lambda s: print(f"  {s}"), label="cmp")


if __name__ == "__main__":
    main()
