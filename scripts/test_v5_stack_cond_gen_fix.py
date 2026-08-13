"""Ad-hoc check: cond generation from a saved qcute_v5_stack/qcute_v5_concat checkpoint, before/
after the generate_no_cache slot-mismatch fix. Uses the docs/status.md same-prompt methodology
(START=5850, PROMPT_LEN=64, GEN_LEN=64, greedy argmax).

    uv run python scripts/test_v5_stack_cond_gen_fix.py <checkpoint.pt> [stack|concat]
"""
import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

START, PROMPT_LEN, GEN_LEN = 5850, 64, 64


def main():
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints/qcute_v5_stack_k1_l1/best.pt")
    variant = sys.argv[2] if len(sys.argv) > 2 else ("concat" if "concat" in str(ckpt_path) else "stack")
    mod = importlib.import_module(f"qcute.qcute_v5_{variant}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = mod.Config(**ckpt["cfg"])
    model = mod.RefineLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_path} ({variant}) step={ckpt.get('step')} Ks={cfg.Ks} n_layers={cfg.n_layers}")

    data = mod.load_enwik8(Path("datasets/enwik8_1M.gz"), n_bytes=10000)
    train_data, _ = mod.split_train_val(data, 0.1)
    window = train_data[START:START + PROMPT_LEN + GEN_LEN]
    prompt = window[:PROMPT_LEN]
    ground_truth = window[PROMPT_LEN:]

    cond = mod.generate_no_cache(model, prompt, GEN_LEN, device)
    uncond = mod.generate_encode_only(model, prompt, GEN_LEN, device)

    print("prompt:      ", bytes(prompt.tolist()))
    print("ground_truth:", bytes(ground_truth.tolist()))
    print("uncond:      ", bytes(uncond[PROMPT_LEN:].tolist()))
    print("cond:        ", bytes(cond[PROMPT_LEN:].tolist()))


if __name__ == "__main__":
    main()
