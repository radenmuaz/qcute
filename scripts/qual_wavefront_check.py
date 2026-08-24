"""Qualitative check for qcute_zero's generate_wavefront (2026-08-24 PoC): loads a trained
checkpoint (configs/qcute_zero/ks1_overfit10k_wavefront2.py's own run) and shows four decode modes
side by side (chat 2026-08-24's "later for wavefront there are more modes" plan):
  - ntp:           generate_no_cache -- the ground-truth reference, plain greedy AR.
  - mtp:           generate_speculative -- MTP-drafted, byte-by-byte VERIFIED against ntp
                    (guaranteed identical to ntp; shown to confirm that guarantee directly).
  - wavefront-ntp: generate_wavefront(n_waves=2) -- unverified, greedy at every lockstep step.
                    This is a genuinely DIFFERENT generative process (own MTP-bootstrapped wave
                    starts, cross-region visibility), so it is NOT expected to match ntp exactly --
                    a plausibility/quality check, not a consistency check.
  - wavefront-mtp: generate_wavefront_mtp -- wavefront-DRAFTED (region_len passes/block), byte-by-byte
                    VERIFIED against the same exact stepper generate_speculative uses (guaranteed
                    identical to ntp; accept_rate expected <= plain mtp's once region_len>1, since
                    only then does the lockstep independence assumption actually bite -- see
                    generate_wavefront_mtp's own docstring).

Also checks the checkpoint's own check_wavefront_consistency (n_waves=1 degenerate case, which
MUST match ntp exactly -- see generate_wavefront's own docstring).

uv run python scripts/qual_wavefront_check.py --checkpoint logs/qcute_zero_ks1_overfit10k_wavefront2/last.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qcute.qcute_zero.qcute_zero import Config, QCuteZero, load_enwik8, pack_words, split_train_val


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--n_bytes", type=int, default=10000)
    p.add_argument("--prompt_len", type=int, default=16)
    p.add_argument("--n_new_bytes", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = Config(**ckpt["cfg"])
    model = QCuteZero(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded checkpoint from step {ckpt['step']}, Ks={cfg.Ks}, mtp_heads={cfg.mtp_heads}")

    data = load_enwik8(args.data, cfg.input_preset, args.n_bytes)
    _, val = split_train_val(data, 0.1)

    consistency = model.check_wavefront_consistency(
        val, args.device, n_checks=5, prompt_len=args.prompt_len, K=8, n_new_bytes=args.n_new_bytes)
    print(f"n_waves=1 degenerate consistency (must be 1.0): {consistency}")

    prompt = val[:args.prompt_len]
    out_ntp = model.generate_no_cache(prompt, args.n_new_bytes, args.device)
    out_mtp, spec_stats = model.generate_speculative(prompt, args.n_new_bytes, args.device, return_stats=True)
    out_wave_ntp = model.generate_wavefront(prompt, K=8, n_waves=2, n_new_bytes=args.n_new_bytes, device=args.device)
    out_wave_mtp, wave_spec_stats = model.generate_wavefront_mtp(
        prompt, K=8, n_waves=2, n_new_bytes=args.n_new_bytes, device=args.device, return_stats=True)

    def to_text(t: torch.Tensor) -> str:
        return pack_words(t.tolist(), cfg.input_preset).decode("latin-1", errors="replace")

    print(f"\nprompt:              {to_text(prompt)!r}")
    print(f"ntp:                 {to_text(out_ntp)!r}")
    print(f"mtp (accept_rate={spec_stats['accept_rate']:.2f}, matches ntp={torch.equal(out_ntp, out_mtp)}): "
          f"{to_text(out_mtp)!r}")
    print(f"wavefront-ntp:       {to_text(out_wave_ntp)!r}")
    print(f"wavefront-mtp (accept_rate={wave_spec_stats['accept_rate']:.2f}, "
          f"matches ntp={torch.equal(out_ntp, out_wave_mtp)}): {to_text(out_wave_mtp)!r}")


if __name__ == "__main__":
    main()
