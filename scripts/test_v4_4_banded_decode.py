"""Verify LevelLM._packed_decode_forward_banded (general multi-track, any K, any decode_pack_mode)
is numerically identical to the dense _packed_decode_forward reference, across single-track K==1,
single-track K>1 (decode_K>1, dense-only territory before this), and multi-track cumulative
cross-level conditioning with per-track windows. Also times both at production scale.

    uv run python scripts/test_v4_4_banded_decode.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qcute.qcute_refine_v4_4 import Config, LevelLM


def make_level(d_model, n_layers, n_heads=4, seed=0):
    torch.manual_seed(seed)
    cfg = Config(Ks=(4,), d_model=d_model, n_layers=n_layers, n_heads=n_heads, vocab=256)
    level = LevelLM(cfg, level=0, window=None, decode_windows=[None])
    level.eval()
    return level


def check_equivalence(label, L, tracks_spec, d_model=32, n_layers=2, n_heads=4,
                       margin_extra_chunks=1, seed=0):
    """tracks_spec: list of (K, window) pairs, finest (self) first, matching RefineLM._run's order."""
    level = make_level(d_model, n_layers, n_heads, seed=seed)
    D = d_model
    B = 2
    torch.manual_seed(seed + 1)
    x0 = torch.randn(B, L, D)
    tracks = []
    for K, window in tracks_spec:
        assert L % K == 0
        n_blocks = L // K
        code_kv = torch.randn(B, n_blocks, D)
        tracks.append((code_kv, K, window))

    with torch.no_grad():
        h_dense = level._packed_decode_forward(x0, tracks)
        h_banded = level._packed_decode_forward_banded(x0, tracks, margin_extra_chunks=margin_extra_chunks)

    max_diff = (h_dense - h_banded).abs().max().item()
    match = torch.allclose(h_dense, h_banded, atol=1e-4, rtol=1e-4)
    print(f"{label}: L={L} tracks={tracks_spec} margin={margin_extra_chunks}: "
          f"max_diff={max_diff:.2e} allclose={match}")
    return match


def time_decode(label, L, tracks_spec, d_model=256, n_layers=2, n_heads=4, device="cpu", n_iters=5):
    level = make_level(d_model, n_layers, n_heads).to(device)
    D = d_model
    B = 4
    x0 = torch.randn(B, L, D, device=device)
    tracks = []
    for K, window in tracks_spec:
        n_blocks = L // K
        tracks.append((torch.randn(B, n_blocks, D, device=device), K, window))
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)

    for fn_name, fn in (("dense", level._packed_decode_forward), ("banded", level._packed_decode_forward_banded)):
        if fn_name == "dense" and L > 1024:
            print(f"{label} {fn_name}: skipped (O((2L)^2) -- expected impractical at L={L})")
            continue
        with torch.no_grad():
            fn(x0, tracks)
            sync()
            t0 = time.time()
            for _ in range(n_iters):
                fn(x0, tracks)
            sync()
        dt = (time.time() - t0) / n_iters
        print(f"{label} {fn_name}: {dt*1000:.1f}ms/iter (L={L}, tracks={tracks_spec}, device={device})")


def main():
    print("=== correctness: banded vs dense, single-track K=1 (matches old chunked-test territory) ===")
    ok = True
    ok &= check_equivalence("K1 tiny margin=0", L=64, tracks_spec=[(1, 8)], margin_extra_chunks=0)
    ok &= check_equivalence("K1 tiny margin=1", L=64, tracks_spec=[(1, 8)], margin_extra_chunks=1)
    print()

    print("=== correctness: banded vs dense, single-track K>1 (decode_K>1, previously dense-only) ===")
    ok &= check_equivalence("K4 W8", L=64, tracks_spec=[(4, 8)], margin_extra_chunks=1)
    ok &= check_equivalence("K4 W2 (narrow)", L=64, tracks_spec=[(4, 2)], margin_extra_chunks=1)
    print()

    print("=== correctness: banded vs dense, multi-track cumulative (self + cross, different K/window) ===")
    ok &= check_equivalence("Ks=(4,1) self+cross same K", L=64, tracks_spec=[(4, 8), (4, 32)], margin_extra_chunks=1)
    ok &= check_equivalence("Ks=(4,2) self K=4, cross K=8", L=64, tracks_spec=[(4, 8), (8, 16)], margin_extra_chunks=1)
    ok &= check_equivalence("3 tracks Ks=(2,2,2)", L=64, tracks_spec=[(2, 4), (4, 8), (8, 16)], margin_extra_chunks=1)
    print()

    print("=== correctness: banded vs dense, production-ish scale ===")
    ok &= check_equivalence("prod K4 W16", L=256, tracks_spec=[(4, 16)], d_model=256, margin_extra_chunks=1)
    ok &= check_equivalence("prod Ks=(4,1) two-track", L=256, tracks_spec=[(4, 8), (4, 64)],
                             d_model=256, margin_extra_chunks=1)
    print()

    if not ok:
        print("FAIL: banded decode does not match dense for some config")
        sys.exit(1)
    print("ALL MATCH")

    print()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== timing: dense vs banded, device={device} ===")
    for L in (256, 1024, 4096):
        time_decode(f"K4 W16 L={L}", L=L, tracks_spec=[(4, 16)], device=device)
        print()
    time_decode("Ks=(4,1) two-track L=1024", L=1024, tracks_spec=[(4, 8), (4, 64)], device=device)


if __name__ == "__main__":
    main()
