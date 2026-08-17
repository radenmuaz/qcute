"""Correctness check for qcute_v5_concat_modes.py's multi_mode_impl:
(1) "off" reproduces plain qcute_v5_concat.py's forward() bit-for-bit (same weights, same input --
    the fork must be a true no-op when multi-mode is disabled).
(2) "single_pass" matches "multipass" exactly across Ks=(1,), (4,1), (2,2,1), BOTH with unbounded
    (dense) windows and with a finite window (chunked/banded path -- exercises
    _merged_decode_forward_multimode_chunked, the batched SWA-compatible implementation added
    2026-08-17) -- same weights, same input, per-mode losses and per-level total loss/metrics
    compared with torch.allclose.
(3) The chunked single_pass path also matches _merged_decode_forward_multimode_looped directly
    (option 1, the simplest possible SWA-compatible reference -- literally T calls to the existing,
    unmodified _merged_decode_forward) as an extra cross-check independent of "multipass"'s own
    LevelLM.forward plumbing.

    uv run python scripts/test_v5_concat_modes.py
"""
import torch

from qcute import qcute_v5_concat as plain
from qcute import qcute_v5_concat_modes as modes

DEVICE = "cpu"
torch.manual_seed(0)


def make_models(Ks, context_len, attn_window=-1, n_layers=2, d_model=32, n_heads=4):
    cfg_plain = plain.Config(Ks=Ks, d_model=d_model, n_layers=n_layers, context_len=context_len,
                              n_heads=n_heads, attn_window=attn_window, vocab=256)
    cfg_modes_off = modes.Config(Ks=Ks, d_model=d_model, n_layers=n_layers, context_len=context_len,
                                  n_heads=n_heads, attn_window=attn_window, vocab=256, multi_mode_impl="off")
    cfg_multipass = modes.Config(Ks=Ks, d_model=d_model, n_layers=n_layers, context_len=context_len,
                                  n_heads=n_heads, attn_window=attn_window, vocab=256, multi_mode_impl="multipass")
    cfg_single = modes.Config(Ks=Ks, d_model=d_model, n_layers=n_layers, context_len=context_len,
                               n_heads=n_heads, attn_window=attn_window, vocab=256, multi_mode_impl="single_pass")

    m_plain = plain.RefineLM(cfg_plain).to(DEVICE).eval()
    m_off = modes.RefineLM(cfg_modes_off).to(DEVICE).eval()
    m_off.load_state_dict(m_plain.state_dict())
    m_multi = modes.RefineLM(cfg_multipass).to(DEVICE).eval()
    m_multi.load_state_dict(m_plain.state_dict())
    m_single = modes.RefineLM(cfg_single).to(DEVICE).eval()
    m_single.load_state_dict(m_plain.state_dict())
    return m_plain, m_off, m_multi, m_single


def test_off_matches_plain(Ks, context_len, attn_window=-1):
    m_plain, m_off, _, _ = make_models(Ks, context_len, attn_window)
    x = torch.randint(0, 256, (2, context_len))
    with torch.no_grad():
        loss_p, metrics_p = m_plain(x)
        loss_o, metrics_o = m_off(x)
    assert torch.allclose(loss_p, loss_o, atol=1e-6), f"off vs plain loss mismatch: {loss_p.item()} vs {loss_o.item()}"
    for k in metrics_p:
        if k not in metrics_o:
            continue
        vp, vo = metrics_p[k], metrics_o[k]
        if torch.is_tensor(vp) and torch.is_tensor(vo):
            assert torch.allclose(vp, vo, atol=1e-5), f"off vs plain metric {k} mismatch: {vp.item()} vs {vo.item()}"
    print(f"  Ks={Ks} ctx={context_len} window={attn_window}: off == plain  OK")


def test_single_pass_matches_multipass(Ks, context_len, attn_window=-1):
    _, _, m_multi, m_single = make_models(Ks, context_len, attn_window)
    x = torch.randint(0, 256, (2, context_len))
    with torch.no_grad():
        loss_m, metrics_m = m_multi(x)
        loss_s, metrics_s = m_single(x)
    ok = torch.allclose(loss_m, loss_s, atol=1e-4)
    print(f"  Ks={Ks} ctx={context_len} window={attn_window}: multipass loss={loss_m.item():.6f}  "
          f"single_pass loss={loss_s.item():.6f}  {'OK' if ok else 'MISMATCH'}")
    assert ok, f"single_pass vs multipass total loss mismatch for Ks={Ks} window={attn_window}"
    for k in ("decode_total", "decode_stage_extra_total", "ntp_loss_total", "byte_loss"):
        vm, vs = metrics_m[k], metrics_s[k]
        assert torch.allclose(vm, vs, atol=1e-4), (
            f"single_pass vs multipass metric {k} mismatch for Ks={Ks} window={attn_window}: {vm.item()} vs {vs.item()}")
    print(f"    all per-level metrics match ({', '.join(('decode_total','decode_stage_extra_total','ntp_loss_total','byte_loss'))})")

    # cross-check: the chunked batched path (option 2) also matches the simplest possible
    # SWA-compatible reference (option 1) directly, independent of LevelLM.forward's own plumbing.
    # Only meaningful for a real finite window (attn_window=-1 means unbounded/None -- skip, the
    # model-level test above already covers that case via the dense block-diagonal path).
    x0 = m_single.encode_lms[0].embed(x)
    decode_tracks = []
    cum_K = 1
    window_meta = attn_window if attn_window is not None and attn_window > 0 else None
    for j, K in enumerate(m_single.cfg.Ks):
        cum_K *= K
        decode_tracks.append((torch.randn(2, context_len // cum_K, m_single.cfg.d_model), cum_K, window_meta))
    if len(decode_tracks) > 1 and window_meta is not None:
        with torch.no_grad():
            h_full_chunked, h_modes_chunked = m_single.decode_lms[0]._merged_decode_forward_multimode(x0, decode_tracks)
            h_full_looped, h_modes_looped = m_single.decode_lms[0]._merged_decode_forward_multimode_looped(x0, decode_tracks)
        assert torch.allclose(h_full_chunked, h_full_looped, atol=1e-4), "chunked vs looped h_full mismatch"
        for hc, hl in zip(h_modes_chunked, h_modes_looped):
            assert torch.allclose(hc, hl, atol=1e-4), "chunked vs looped h_modes mismatch"
        print("    chunked (option 2) == looped (option 1) direct cross-check  OK")


if __name__ == "__main__":
    print("=== off == plain (no-op check) ===")
    for Ks, ctx in [((1,), 16), ((4, 1), 32), ((2, 2, 1), 32)]:
        test_off_matches_plain(Ks, ctx)

    print("\n=== single_pass == multipass (dense, unbounded window) ===")
    for Ks, ctx in [((1,), 16), ((4, 1), 32), ((2, 2, 1), 32)]:
        test_single_pass_matches_multipass(Ks, ctx)

    print("\n=== single_pass == multipass (chunked/banded, finite window) ===")
    for Ks, ctx, window in [((1,), 16, 4), ((4, 1), 32, 8), ((2, 2, 1), 32, 8)]:
        test_single_pass_matches_multipass(Ks, ctx, window)

    print("\nAll checks passed.")
