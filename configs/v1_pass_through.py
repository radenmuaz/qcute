"""qcute.qcute_refine_v2 config: CLONE of configs/v1_rope.py (itself a
clone of the "v1" baseline, configs/qcute_refine_v2_byte4_code256_simple.py),
two changes — decoder_kv_pass_through=True AND decoder_q_pass_through=True.

BOTH sides of DecoderLevel's cross-attention now bypass EncoderLevel's
own hidden states entirely:
  - KV comes from a fresh Linear(dqs[level], tok_d_model) projection of
    the level's own raw emitted code c_i, instead of
    EncoderLevel[level+1]'s hidden state.
  - Q comes from a fresh direct embedding of this level's own raw input
    seq_repr (nn.Embedding(vocab, tok_d_model) since byte_repr="embed"
    here), instead of EncoderLevel[level]'s hidden state h_prev.

Session ask, verbatim reason: "use q embed not h... reason is to see
limits of decoder" — with zero causal self-attention context on EITHER
side, this is a deliberate floor/worst-case probe: how well can plain
cross-attention alone do, given only raw per-position embeddings and a
linearly-projected code, no contextualization at all? See
qcute_refine_v2.py's Config.decoder_q_pass_through docstring.

QUEUED — do not launch until the "v1" baseline
(qcute_refine_v2_byte4_code256_simple) finishes; do not touch that
config or its run.

    uv run python -m qcute.qcute_refine_v2 --config configs/v1_pass_through.py

    # plot after training:
    uv run python scripts/plot_run.py logs/v1_pass_through
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
decoder_kv_pass_through = True
decoder_q_pass_through = True

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
