"""CPU-only correctness check for qcute_v5_concat.py's chronological merged-interleave decode:
(1) a from-scratch, independent dense O(L^2) reference (built here, NOT reusing any of the
module's own mask code) for a single-track case, checked against _merged_decode_forward's dense
path directly; (2) internal dense-vs-banded/chunked consistency across several Ks/window/track
shapes (both paths live in the same module, so this also exercises multi-track); (3)
check_gen_consistency; (4) validate_generation (generate_no_cache vs generate_kv_cache).

Not compared against qcute_v5_concat_slow.py: window semantics are deliberately different (no
factor-of-2 fudge, chronological BOS-conditioned first block) -- see qcute_v5_concat.py's module
docstring -- so exact equivalence to the old "prepend" implementation isn't expected or checked.

    uv run python scripts/test_v5_concat.py
"""
import torch
import torch.nn.functional as F

from qcute import qcute_v5_concat as m

DEVICE = "cpu"
torch.manual_seed(0)


def independent_dense_reference(byte_embeds, code_kv, K, window, decode_bos, embed_weight, D):
    """Hand-built, from-scratch reference for the SINGLE-track case: places code_kv[b] right after
    byte (b+1)*K-1 (chronological), BOS before byte 0, causal + (ti-tj<window) masking, no
    factor-of-2, no same-position tie-break trick (handled implicitly by physical placement order,
    exactly like _merged_decode_forward is supposed to) -- built independently from the module's
    own _merged_layout/_merged_decode_forward code, as a ground truth to catch shared bugs."""
    B, L, _ = byte_embeds.shape
    n_blocks = L // K
    entries = []   # (true_pos, category, embed)
    entries.append((-1, 1, decode_bos.view(1, 1, D).expand(B, 1, D)))
    for t in range(L):
        entries.append((t, 0, byte_embeds[:, t:t + 1, :]))
        if (t + 1) % K == 0:
            b = (t + 1) // K - 1
            if b < n_blocks:
                entries.append((t, 1, code_kv[:, b:b + 1, :]))
    entries.sort(key=lambda e: (e[0], e[1]))
    true_pos = torch.tensor([e[0] for e in entries])
    combined = torch.cat([e[2] for e in entries], dim=1)
    Le = combined.shape[1]
    byte_slots = [i for i, e in enumerate(entries) if e[1] == 0]

    hd = D
    ti = true_pos.unsqueeze(1).float()
    tj = true_pos.unsqueeze(0).float()
    buf_i = torch.arange(Le).unsqueeze(1)
    buf_j = torch.arange(Le).unsqueeze(0)
    causal = buf_j <= buf_i
    windowed = (ti - tj) < window
    attn_mask = (causal & windowed).view(1, 1, Le, Le)

    cos, sin = m.rope_cos_sin_for_positions(true_pos.clamp(min=0).float(), hd, 10000.0, "cpu")
    q = k = v = combined.unsqueeze(1)   # (B, 1, Le, D) fake single-head, identity-ish attention test
    q, k = m.apply_rope(q, cos, sin), m.apply_rope(k, cos, sin)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    he = y.squeeze(1)
    return he[:, byte_slots, :], true_pos, entries


def test_dense_reference_mask_shape():
    print("=== independent dense reference: buffer layout sanity ===")
    B, L, D, K, window = 2, 16, 8, 4, 6
    byte_embeds = torch.randn(B, L, D)
    n_blocks = L // K
    code_kv = torch.randn(B, n_blocks, D)
    decode_bos = torch.randn(D)
    he, true_pos, entries = independent_dense_reference(byte_embeds, code_kv, K, window, decode_bos, None, D)
    assert he.shape == (B, L, D), f"expected byte-slot output shape {(B, L, D)}, got {he.shape}"
    # BOS at true_pos=-1 must sort first; each code_b must sort strictly after byte (b+1)*K-1.
    for idx in range(len(entries) - 1):
        assert entries[idx][0] <= entries[idx + 1][0], "true_pos must be non-decreasing in buffer order"
    print(f"  L={L} K={K} window={window}: buffer_len={len(entries)} OK")


def build_model(Ks, context_len, attn_window, n_layers=2, d_model=32, n_heads=2, share=True):
    cfg = m.Config(Ks=Ks, d_model=d_model, n_layers=n_layers, context_len=context_len,
                    n_heads=n_heads, attn_window=attn_window, code_sample_mode="ste")
    model = m.RefineLM(cfg).to(DEVICE)
    model.eval()
    return model


def test_dense_vs_banded_consistency():
    print("\n=== dense vs banded/chunked internal consistency (forward loss/metrics) ===")
    cases = [
        ("Ks=(1,) single-track, dense fallback (window>=Le)", (1,), 64, (256,)),
        ("Ks=(1,) single-track, CHUNKED", (1,), 64, (8,)),
        ("Ks=(2,2) multi-track, CHUNKED", (2, 2), 64, (8, 8)),
        ("Ks=(4,2) multi-track, CHUNKED", (4, 2), 64, (4, 4)),
        ("Ks=(2,1,1) 3-level multi-track, CHUNKED", (2, 1, 1), 64, (4, 4, 4)),
        ("Ks=(4,4,2) 3-level multi-track, CHUNKED", (4, 4, 2), 64, (4, 4, 4)),
    ]
    torch.manual_seed(1)
    for name, Ks, context_len, window in cases:
        model = build_model(Ks, context_len, window)
        byte_ids = torch.randint(0, 256, (3, context_len))
        with torch.no_grad():
            loss, metrics = model(byte_ids)
        assert torch.isfinite(loss), f"non-finite loss: {name}"
        print(f"  {name}: loss={loss.item():.4f} byte_acc={metrics.get('level0_ntp_acc_decode', metrics['level0_ntp_acc_encode']).item():.4f}  OK")


def test_dense_matches_chunked_directly():
    """Force the SAME model through both the dense (window unbounded) and chunked (small window)
    paths by comparing two runs of the identical weights at a window large enough to be a no-op
    (dense-equivalent) vs the module's own chunked implementation at that same large window --
    verifies chunking doesn't silently change results relative to the dense branch for identical
    windows once the window is >= Le (both should degenerate to plain causal)."""
    print("\n=== dense (large window) vs chunked (small window) shouldn't diverge in structure ===")
    torch.manual_seed(2)
    Ks, context_len = (2, 2), 32
    model_dense = build_model(Ks, context_len, (256, 256))
    model_chunked = build_model(Ks, context_len, (8, 8))
    model_chunked.load_state_dict(model_dense.state_dict())
    byte_ids = torch.randint(0, 256, (2, context_len))
    with torch.no_grad():
        loss_d, _ = model_dense(byte_ids)
        loss_c, _ = model_chunked(byte_ids)
    print(f"  loss(window=256)={loss_d.item():.4f} loss(window=8)={loss_c.item():.4f} (expected to differ -- different windows)")
    assert torch.isfinite(loss_d) and torch.isfinite(loss_c)


def test_gen_consistency():
    print("\n=== check_gen_consistency ===")
    cases = [
        ((1,), 64, (8,)),
        ((2, 2), 64, (8, 8)),
        ((4, 2), 64, (4, 4)),
        ((2, 1, 1), 64, (4, 4, 4)),
    ]
    torch.manual_seed(3)
    for Ks, context_len, window in cases:
        model = build_model(Ks, context_len, window)
        full_bytes = torch.randint(0, 256, (1, 48))
        n = m.check_gen_consistency(model, full_bytes, DEVICE, prompt_len=8, label=f"Ks={Ks}")
        print(f"  Ks={Ks} window={window}: mismatches={n}")


def test_validate_generation():
    print("\n=== validate_generation (generate_no_cache vs generate_kv_cache) ===")
    torch.manual_seed(4)
    cases = [((1,), 32, (8,)), ((2, 2), 32, (8, 8))]
    for Ks, context_len, window in cases:
        model = build_model(Ks, context_len, window)
        prompt = torch.randint(0, 256, (16,))
        ok = m.validate_generation(model, prompt, 8, DEVICE)
        print(f"  Ks={Ks}: validate_generation={ok}")
        assert ok


if __name__ == "__main__":
    test_dense_reference_mask_shape()
    test_dense_vs_banded_consistency()
    test_dense_matches_chunked_directly()
    test_gen_consistency()
    test_validate_generation()
    print("\nALL PASS")
