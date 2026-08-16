"""CPU-only correctness check for qcute_v5.py (stack decode, efficient windowed attention)
against the original qcute_v5_stack.py: (1) chunked_windowed_attention vs a brute-force dense
reference, (2) full RefineLM forward equivalence with copied weights (orig vs eff) -- exercising
both the ported selfcode_decode chunked path and the new cross_attn_stage chunked path,
(3) check_gen_consistency on the eff model.

    uv run python scripts/test_v5.py
"""
import torch
import torch.nn.functional as F

from qcute import qcute_v5_stack as orig
from qcute import qcute_v5 as eff

DEVICE = "cpu"
torch.manual_seed(0)


def dense_windowed_attention(q, k, v, window):
    B, H, T, hd = q.shape
    pos = torch.arange(T)
    ti, tj = pos.unsqueeze(1), pos.unsqueeze(0)
    attn_mask = ((tj <= ti) & (ti - tj < window)).view(1, 1, T, T)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


def test_chunked_windowed_attention():
    print("=== chunked_windowed_attention vs dense reference ===")
    for T, window, hd in [(64, 8, 16), (32, 32, 8), (128, 16, 4), (48, 5, 8)]:
        B, H = 2, 3
        q = torch.randn(B, H, T, hd)
        k = torch.randn(B, H, T, hd)
        v = torch.randn(B, H, T, hd)
        got = eff.chunked_windowed_attention(q, k, v, window)
        want = dense_windowed_attention(q, k, v, window)
        max_diff = (got - want).abs().max().item()
        status = "OK" if max_diff < 1e-4 else "FAIL"
        print(f"  T={T:4d} window={window:3d} hd={hd:3d}: max_diff={max_diff:.2e}  {status}")
        assert max_diff < 1e-4, f"chunked_windowed_attention mismatch at T={T} window={window}"


def copy_weights(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    dst.load_state_dict(src.state_dict())


def build_pair(Ks, context_len, attn_window, n_layers=2, d_model=32, n_heads=2):
    cfg_kwargs = dict(
        Ks=Ks, d_model=d_model, n_layers=n_layers, context_len=context_len, n_heads=n_heads,
        attn_window=attn_window, use_gumbel_noise=False, share_level_weights=True,
    )
    # eff.Config hardcodes decode_code_ste=True, cross_track_source="decode", and has no
    # decode_self_only_aux -- set orig's equivalents explicitly so both sides match.
    cfg_o = orig.Config(**cfg_kwargs, decode_self_only_aux=False, decode_code_ste=True,
                         cross_track_source="decode")
    cfg_e = eff.Config(**cfg_kwargs)
    m_o = orig.RefineLM(cfg_o).to(DEVICE)
    m_e = eff.RefineLM(cfg_e).to(DEVICE)
    copy_weights(m_o, m_e)
    m_o.eval()
    m_e.eval()
    return m_o, m_e


def test_forward_equivalence():
    print("\n=== full-model forward equivalence (orig vs eff, copied weights) ===")
    cases = [
        ("Ks=(1,) single-track selfcode, dense fallback (window>=Le)", (1,), 64, (256,)),
        ("Ks=(1,) single-track selfcode, CHUNKED", (1,), 64, (8,)),
        ("Ks=(2,2) level0 multi-track cross_attn_stage CHUNKED + level1 selfcode CHUNKED", (2, 2), 64, (8, 8)),
        ("Ks=(4,2) level0 multi-track cross_attn_stage CHUNKED + level1 selfcode CHUNKED", (4, 2), 64, (4, 4)),
        ("Ks=(2,1,1) 3-level: multi-track cross_attn_stage + single-track chunked", (2, 1, 1), 64, (4, 4, 4)),
    ]
    torch.manual_seed(1)
    for name, Ks, context_len, window in cases:
        m_o, m_e = build_pair(Ks, context_len, window)
        byte_ids = torch.randint(0, 256, (3, context_len))
        with torch.no_grad():
            loss_o, metrics_o = m_o(byte_ids)
            loss_e, metrics_e = m_e(byte_ids)
        loss_diff = (loss_o - loss_e).abs().item()
        common_keys = metrics_o.keys() & metrics_e.keys()
        max_metric_diff = max((metrics_o[k] - metrics_e[k]).abs().item() for k in common_keys)
        status = "OK" if loss_diff < 1e-4 and max_metric_diff < 1e-4 else "FAIL"
        print(f"  {name}: loss_diff={loss_diff:.2e} max_metric_diff={max_metric_diff:.2e}  {status}")
        assert loss_diff < 1e-4 and max_metric_diff < 1e-4, f"forward mismatch: {name}"


def test_gen_consistency():
    print("\n=== check_gen_consistency on eff model ===")
    cases = [
        ("Ks=(1,)", (1,), 64, (8,)),
        ("Ks=(2,2)", (2, 2), 64, (8, 8)),
        ("Ks=(4,2)", (4, 2), 64, (4, 4)),
        ("Ks=(2,1,1)", (2, 1, 1), 64, (4, 4, 4)),
    ]
    torch.manual_seed(2)
    for name, Ks, context_len, window in cases:
        _, m_e = build_pair(Ks, context_len, window)
        full_bytes = torch.randint(0, 256, (1, context_len))
        n_mismatch = eff.check_gen_consistency(m_e, full_bytes, DEVICE, prompt_len=context_len // 2, label=name)
        assert n_mismatch == 0, f"gen consistency FAILED for {name}: {n_mismatch} mismatches"


if __name__ == "__main__":
    test_chunked_windowed_attention()
    test_forward_equivalence()
    test_gen_consistency()
    print("\nALL PASS")
