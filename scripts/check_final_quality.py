"""Post-training quality check against a fixed, saved prompt offset, covering every generation
mode a checkpoint supports (uncond/encode-only, decode self, and -- for n_levels>1 --
decode self+coarser/cond_full), via each module's own qualitative_generate + check_gen_consistency.

    uv run python scripts/check_final_quality.py concat <checkpoint.pt>
    uv run python scripts/check_final_quality.py stack <checkpoint.pt>
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

START, PROMPT_LEN, GEN_LEN = 100, 64, 64


def main():
    variant = sys.argv[1]
    ckpt_path = Path(sys.argv[2])
    assert variant in ("concat", "stack"), "first arg must be 'concat' or 'stack'"

    if variant == "concat":
        from qcute.qcute_v5_concat import (
            Config, RefineLM, qualitative_generate, check_gen_consistency, load_enwik8, split_train_val,
        )
    else:
        from qcute.qcute_v5_stack import (
            Config, RefineLM, qualitative_generate, check_gen_consistency, load_enwik8, split_train_val,
        )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Config(**ckpt["cfg"])
    model = RefineLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_path} step={ckpt.get('step')} Ks={cfg.Ks} n_layers={cfg.n_layers} "
          f"best_val_bpb={ckpt.get('val_bpb', '?')}")

    data = load_enwik8(Path("datasets/enwik8_1M.gz"), n_bytes=10000)
    train_data, val_data = split_train_val(data, 0.1)
    for label, src in (("train", train_data), ("val", val_data)):
        window = src[START:START + PROMPT_LEN + GEN_LEN] if label == "train" else src[:PROMPT_LEN + GEN_LEN]
        prompt = window[:PROMPT_LEN]
        gt = window[PROMPT_LEN:]
        qualitative_generate(model, prompt, GEN_LEN, gt, device, log=print, label=label)
        check_gen_consistency(model, window, device, prompt_len=PROMPT_LEN, log=print, label=label)


if __name__ == "__main__":
    main()
