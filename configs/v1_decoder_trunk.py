"""qcute.qcute_refine_v2 config: CLONE of configs/v1_rope.py (itself a
clone of the "v1" baseline, configs/qcute_refine_v2_byte4_code256_simple.py),
one change — decoder_own_trunk=True.

DecoderLevel builds its OWN separate-weight copies of EncoderLevel[level]/
[level+1]'s own trunk shape (via two private EncoderLevel instances) and
runs raw sequences through them itself, instead of reusing
EncoderLevel's own already-computed hidden states (h_prev/h_curr) — the
"own trunk" design discussed this session (see qcute_refine_v2.py's
Config.decoder_own_trunk docstring). Session FLOPs/param estimate for
this exact swap: ~+61% params, ~+57% FLOPs on the decoder specifically
(+1.65M params, +2.58 GFLOPs fwd) versus the reuse-h default.

Question this run answers: does giving each DecoderLevel its own
separate-weight trunk (rather than sharing/reusing the encoder's) improve
prediction quality enough to justify that cost — or does reuse already
capture what's needed (per the session's own probe_decoder_kv_contribution.py
diagnostic direction)?

QUEUED — do not launch until the "v1" baseline
(qcute_refine_v2_byte4_code256_simple) finishes; do not touch that
config or its run.

    uv run python -m qcute.qcute_refine_v2 --config configs/v1_decoder_trunk.py

    # plot after training:
    uv run python scripts/plot_run.py logs/v1_decoder_trunk
"""
from pathlib import Path

Ks = (4, 4)
dqs = (8, 8)
tier_d_models = (256, 256)
tier_n_layers = (1, 1)
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (256, 64)

byte_repr = "embed"
code_head_mode = "independent"
cross_attn_rope = True
decoder_own_trunk = True

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
