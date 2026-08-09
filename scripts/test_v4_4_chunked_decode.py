"""Verify LevelLM._packed_decode_forward_chunked (interleave mode) is numerically identical to
the dense _packed_decode_forward, and time both at production scale.

    uv run python scripts/test_v4_4_chunked_decode.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qcute.qcute_refine_v4_4 import Config, RefineLM


def check_equivalence(Ks, d_model, n_layers, context_len, window, n_prev_chunks, n_heads=4, seed=0):
    torch.manual_seed(seed)
    cfg = Config(Ks=Ks, d_model=d_model, n_layers=n_layers, context_len=context_len,
                 n_heads=n_heads, attn_window=window, decode_pack_mode="interleave")
    model = RefineLM(cfg)
    model.eval()

    L = model.seq_lens[0]
    x = torch.randint(0, cfg.vocab, (2, context_len))

    with torch.no_grad():
        _, _, _, _, h_dense = model._run(x, compute_ntp=False)
        cfg.decode_chunked = False

    enc0 = model.encoders[0]

    # manually reproduce what RefineLM._run does for level 0's decode pass, but call both
    # dense and chunked paths on the SAME code_kv input for a direct A/B comparison.
    with torch.no_grad():
        c_list = []
        seq_repr = x
        for i in range(model.n_levels):
            c_i, _, _, _ = model.encoders[i](seq_repr, compute_ntp=False)
            c_list.append(c_i)
            seq_repr = c_i
        if model.n_levels > 1:
            source_c = c_list[1]
        else:
            source_c = c_list[0]
        code_ids = source_c.argmax(-1)
        code_embeds = enc0.embed(code_ids)

        x0 = enc0.embed(x)
        h_dense_out = enc0._packed_decode_forward(x0, [(code_embeds, 1, window)])
        h_chunked_out = enc0._packed_decode_forward_chunked(x0, code_embeds, window, n_prev_chunks=n_prev_chunks)

    max_diff = (h_dense_out - h_chunked_out).abs().max().item()
    match = torch.allclose(h_dense_out, h_chunked_out, atol=1e-4, rtol=1e-4)
    print(f"Ks={Ks} L={L} window={window} n_prev_chunks={n_prev_chunks}: "
          f"max_diff={max_diff:.2e} allclose={match}")
    return match


def time_forward(cfg: Config, label: str, device: str, n_iters: int = 3, train_step: bool = True):
    torch.manual_seed(0)
    model = RefineLM(cfg).to(device)
    x = torch.randint(0, cfg.vocab, (4, cfg.context_len), device=device)
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)

    if train_step:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss, _ = model(x)
        loss.backward()
        opt.step()
        opt.zero_grad()
        sync()
        t0 = time.time()
        for _ in range(n_iters):
            loss, _ = model(x)
            loss.backward()
            opt.step()
            opt.zero_grad()
        sync()
    else:
        model.eval()
        with torch.no_grad():
            model(x)
            sync()
            t0 = time.time()
            for _ in range(n_iters):
                model(x)
            sync()
    dt = (time.time() - t0) / n_iters
    print(f"{label}: {dt*1000:.1f}ms/iter (context_len={cfg.context_len}, window={cfg.attn_window}, device={device})")
    return dt


def main():
    print("=== correctness: chunked vs dense (small scale, sweeping n_prev_chunks margin) ===")
    check_equivalence(Ks=(1,), d_model=32, n_layers=2, context_len=64, window=8, n_prev_chunks=1)  # expected insufficient
    ok = True
    for n_prev in (2, 3):
        ok &= check_equivalence(Ks=(1,), d_model=32, n_layers=2, context_len=64, window=8, n_prev_chunks=n_prev)
    print()
    print("=== correctness: chunked vs dense (Ks=(1,1), production-ish scale) ===")
    ok &= check_equivalence(Ks=(1, 1), d_model=64, n_layers=2, context_len=128, window=16, n_prev_chunks=2)
    print()
    print("=== correctness: chunked vs dense (production scale, n_prev_chunks=2) ===")
    ok &= check_equivalence(Ks=(1,), d_model=256, n_layers=2, context_len=256, window=32, n_prev_chunks=2)
    print()
    if not ok:
        print("FAIL: chunked decode does not match dense for some config")
        sys.exit(1)
    print("ALL MATCH")

    print()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== timing (train step: fwd+bwd+opt.step): dense vs chunked, device={device} ===")
    for context_len in (256, 512, 1024):
        cfg_dense = Config(Ks=(1,), d_model=256, n_layers=2, context_len=context_len,
                            n_heads=4, attn_window=32, decode_pack_mode="interleave", decode_chunked=False)
        cfg_chunked = Config(Ks=(1,), d_model=256, n_layers=2, context_len=context_len,
                              n_heads=4, attn_window=32, decode_pack_mode="interleave", decode_chunked=True)
        time_forward(cfg_chunked, f"chunked context_len={context_len}", device)
        if context_len <= 512:
            time_forward(cfg_dense, f"dense   context_len={context_len}", device)
        else:
            print(f"dense   context_len={context_len}: skipped (O((2L)^2) -- expected impractical)")
        print()


if __name__ == "__main__":
    main()
