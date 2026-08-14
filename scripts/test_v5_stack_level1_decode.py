"""Ad-hoc: run full qualitative_generate (incl. the new level1_gen_decode diagnostic) against a
saved qcute_v5_stack checkpoint, using the docs/status.md same-prompt methodology.

    uv run python scripts/test_v5_stack_level1_decode.py <checkpoint.pt>
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qcute.qcute_v5_stack import (
    Config, RefineLM, qualitative_generate, load_enwik8, split_train_val,
)

START, PROMPT_LEN, GEN_LEN = 5850, 64, 64


def main():
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints/qcute_v5_stack_1_baseline/last.pt")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Config(**ckpt["cfg"])
    model = RefineLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_path} step={ckpt.get('step')} Ks={cfg.Ks} n_layers={cfg.n_layers}")

    data = load_enwik8(Path("datasets/enwik8_1M.gz"), n_bytes=10000)
    train_data, val_data = split_train_val(data, 0.1)
    for label, src in (("train", train_data), ("val", val_data)):
        window = src[START:START + PROMPT_LEN + GEN_LEN] if label == "train" else src[:PROMPT_LEN + GEN_LEN]
        prompt = window[:PROMPT_LEN]
        gt = window[PROMPT_LEN:]
        qualitative_generate(model, prompt, GEN_LEN, gt, device, log=print, label=label)


if __name__ == "__main__":
    main()
