"""qcute.qcute_refine_v4 config: IDENTICAL to configs/qcute_refine_v4_
k32_narrow_nonull.py (pre + no null KV) — this file exists only to be
trained under `qcute_refine_v4.py`'s new (this session) live-tracked
"unconditional_pass1" metrics, not to change any architecture/
hyperparameter. See configs/qcute_refine_v4_k32_narrow_postfuse_nonull_
uncond.py's own docstring for the full rationale (its "post" counterpart)
— this is the "pre" half of the same pair, completing both no-null grid
cells under live tracking.

This run recovers `qcute_refine_v4_k32_narrow_nonull`'s own full
trajectory of `val_bpb`/`val_bpb_unconditional` side by side (was a
single post-hoc probe number at step 2700 before: normal 2.5217,
unconditional_pass1 5.0124, docs/kv_contribution.md §10) — lets that
comparison be read directly off one run.jsonl instead of cross-
referencing a separate probe run against a different (though
architecturally identical) trained checkpoint.

Everything else identical to qcute_refine_v4_k32_narrow_nonull.py:
Ks=(32,32), dqs=(8,8), tier_d_models=(256,256), tier_n_layers=(1,1),
context_len=1024, attn_window=(32,32), byte_repr="embed", code_head_mode=
"independent", cross_attn_rope=True, fuse_encoder_levels=True,
code_embed_mode="pq_table", fuse_position="pre" (default),
fuse_use_null_kv=False, steps=4000.

    uv run python -m qcute.qcute_refine_v4 --config configs/qcute_refine_v4_k32_narrow_nonull_uncond.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_k32_narrow_nonull_uncond
"""
from pathlib import Path

Ks = (32, 32)
dqs = (8, 8)
tier_d_models = (256, 256)
tier_n_layers = (1, 1)
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (32, 32)

byte_repr = "embed"
code_head_mode = "independent"
cross_attn_rope = True
fuse_encoder_levels = True
code_embed_mode = "pq_table"
fuse_use_null_kv = False   # fuse_position stays "pre" (default)

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 4000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 50
eval_every = 100
eval_batches = 20
