"""qcute.qcute_refine_v2 config: CLONE of configs/qcute_refine_rope.py,
attn_window changed (256, 64) -> (8, -1) — level 0 (byte encoder) window
shrunk to an EXTREMELY tiny 8 raw bytes, level 1 (code encoder) set to -1
(dense/full attention over its own 256 code positions, i.e. its own
"effective len" now spans the WHOLE 1024-byte context, not just the
64-position/256-byte-equivalent window qcute_refine_rope gave it).

Reason (session ask): "force cross attention to be useful". With
attn_window=8, EncoderLevel[0]'s own self-attention literally cannot see
more than 8 bytes of local history — any longer-range signal the byte-
level NTP head benefits from can ONLY reach it via DecoderLevel[0]'s
cross-attention KV (EncoderLevel[1]'s own hidden states, which DO have
full 1024-byte effective context via level 1's now-dense attention). If
the earlier "simple"/"pass_through" probes' inconclusive-to-mixed KV
contribution finding was partly an artifact of level 0 already having
plenty of local context to lean on instead (attn_window=256 in
qcute_refine_rope already covers a full 256-byte lookback, most of a
4-byte-block's own immediate neighborhood), this config removes that
alternative almost entirely — level 0 is nearly a bag-of-8-bytes model on
its own, so if cross-attention KV is doing anything real, its effect
should show up starkly here (bigger delta_loss_from_kv/delta_acc_from_kv
in scripts/probe_decoder_kv_contribution.py, lower null_slot_attn_mass)
versus if it's still small, that's much stronger evidence the mechanism
itself isn't pulling its weight, not just under-incentivized.

Everything else identical to qcute_refine_rope.py (byte_repr="embed",
code_head_mode="independent", cross_attn_rope=True, tok_head_mode=
"linear", steps=4000).

    uv run python -m qcute.qcute_refine_v2 --config configs/qcute_refine_tiny_byte_window.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_tiny_byte_window

    # then re-run the KV contribution probe against its best checkpoint:
    uv run python scripts/probe_decoder_kv_contribution.py \\
        --checkpoint checkpoints/qcute_refine_tiny_byte_window/best.pt --n_samples 8
"""
from pathlib import Path

Ks = (4, 4)
dqs = (8, 8)
tier_d_models = (256, 256)
tier_n_layers = (1, 1)
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (8, -1)   # level 0: extremely tiny (8 raw bytes). level 1: dense (-1), full 256-position
                         # attention -> effective len covers the WHOLE 1024-byte context.

byte_repr = "embed"
code_head_mode = "independent"
cross_attn_rope = True

tok_d_model = 256
tok_n_heads = 4
tok_mlp_mult = 4
tok_head_mode = "linear"
tok_weight = 1.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 4000   # cut from 8000 (session finding: both bytelm_xs_mtp4_ctx1024 and bpelm_32768 hit
                # their own best val_bpb well before step 2000 and are fully overfit/plateaued by
                # step 4000 — running past that just burns wall-clock without adding comparison signal)
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 50
eval_every = 100
eval_batches = 20
