"""Instrument the REAL `_packed_decode_forward` (2+-track branch, used by cond_full) on a tiny toy
sequence and print its exact causal-mask bookkeeping (true_pos / is_code / windowing) for manual
verification -- not a reimplementation, calls the actual production method via return_debug=True
(qcute_v5_concat.py's LevelLM._packed_decode_forward).

Checks, per byte query position i (true_pos=i):
  - can it see the code key(s) at true_pos=i-1 (its "own" block's prefix, from every track)?
  - is every code/byte key with true_pos >= i correctly excluded (no future leakage)?
  - does windowing cut off keys further than `window` bytes back?

    uv run python scripts/debug_v5_concat_packed_decode_mask.py <checkpoint.pt>
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qcute.qcute_v5_concat import Config, RefineLM


def main():
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints/qcute_v5_concat_1_verify/last.pt")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Config(**ckpt["cfg"])
    model = RefineLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_path} Ks={cfg.Ks} decode_windows={model.decode_windows}")

    decode0 = model.decode_lms[0]
    D = cfg.d_model
    L = 8  # toy byte sequence length
    K_self, window_self = 1, 4
    K_coarse, window_coarse = 1, 4

    x0 = torch.randn(1, L, D, device=device)
    self_track = (torch.randn(1, L // K_self, D, device=device), K_self, window_self)
    coarse_track = (torch.randn(1, L // K_coarse, D, device=device), K_coarse, window_coarse)
    tracks = [self_track, coarse_track]

    with torch.no_grad():
        _, debug = decode0._packed_decode_forward(x0, tracks, return_debug=True)

    true_pos = debug["true_pos"].tolist()
    is_code = debug["is_code"].tolist()
    attn_mask = debug["attn_mask"]  # (Le, Le) bool
    n_prefix = debug["n_prefix"]
    Le = debug["Le"]

    print(f"\nLe={Le} n_prefix={n_prefix} L={L}  (n_prefix should be 2*L/K since 2 tracks, K=1 each -> {2*L})")
    print(f"true_pos: {true_pos}")
    print(f"is_code:  {[int(x) for x in is_code]}")

    # First n_prefix//2 entries are the FIRST-concatenated track's prefixes (coarse, after the
    # internal [::-1] reversal), next n_prefix//2 are the second (self) -- label purely from
    # structural position, not re-deriving the reversal logic.
    half = n_prefix // 2
    track_label = (["track_1st(idx<%d)" % half] * half + ["track_2nd(idx<%d)" % n_prefix] * half
                   + ["byte"] * L)

    print("\nidx  true_pos  is_code  label")
    for idx in range(Le):
        print(f"{idx:3d}  {true_pos[idx]:8d}  {is_code[idx]!s:7}  {track_label[idx]}")

    print("\n--- per-byte-query visibility check ---")
    ok = True
    for b in range(L):
        qidx = n_prefix + b
        assert true_pos[qidx] == b
        visible = [k for k in range(Le) if attn_mask[qidx, k].item()]
        visible_true_pos = sorted(set(true_pos[k] for k in visible))
        own_code_keys = [k for k in visible if is_code[k] and true_pos[k] == b - 1]
        future_leak = [k for k in visible if true_pos[k] >= b and k != qidx]
        print(f"byte i={b} (true_pos={b}): sees true_pos={visible_true_pos}  "
              f"own-block code keys visible={len(own_code_keys)}/2 (expect 2 if b>=1, 0 if b==0)  "
              f"future_leak={future_leak}")
        if b >= 1 and len(own_code_keys) != 2:
            print(f"  !!! byte {b} does NOT see both tracks' code for its own block (true_pos={b-1})")
            ok = False
        if future_leak:
            print(f"  !!! byte {b} leaks future position(s): {future_leak}")
            ok = False

    print(f"\n{'PASS' if ok else 'FAIL'}: own-block-code visibility + no-future-leak checks")


if __name__ == "__main__":
    main()
