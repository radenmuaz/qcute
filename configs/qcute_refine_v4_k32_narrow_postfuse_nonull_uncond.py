"""qcute.qcute_refine_v4 config: IDENTICAL to configs/qcute_refine_v4_
k32_narrow_postfuse_nonull.py (post + no null KV) — this file exists only
to be trained under `qcute_refine_v4.py`'s new (this session)
live-tracked "unconditional_pass1" metrics, not to change any
architecture/hyperparameter.

Session rationale: every `unconditional_pass1` number reported so far
(docs/kv_contribution.md §7-10) came from `scripts/probe_v4_fusion_
contribution.py`, a POST-HOC script re-evaluating a finished checkpoint —
a single number at the best-checkpoint step, not a trajectory. `RefineLM.
_encode`/`forward`/`eval_model` now compute and log this every eval step
for free (PASS 1's own per-level loss/acc was always computed already —
it was just being silently overwritten by PASS 2 before this session's
change; see the module's own `_encode` docstring): new metrics
`byte_loss_unconditional`, `bpb_unconditional`, `level{i}_ntp_loss_
unconditional`, `level{i}_ntp_acc_unconditional` (val-prefixed in
run.jsonl at eval steps, e.g. `val_bpb_unconditional`). `level1` (the top,
never-fused level) trivially equals its own regular value every step —
kept for a uniform per-level column, not informative on its own.

This run recovers `qcute_refine_v4_k32_narrow_postfuse_nonull`'s own
full trajectory of `val_bpb`/`val_bpb_unconditional` side by side (was a
single post-hoc number at step 1800 before: normal 2.5005, unconditional_
pass1 5.6564, docs/kv_contribution.md §9) — lets that comparison be read
directly off one run.jsonl instead of cross-referencing a separate probe
run against the OLD checkpoint (a different, though architecturally
identical, trained model — same config, different random init/run).

Everything else identical to qcute_refine_v4_k32_narrow_postfuse_nonull.py:
Ks=(32,32), dqs=(8,8), tier_d_models=(256,256), tier_n_layers=(1,1),
context_len=1024, attn_window=(32,32), byte_repr="embed", code_head_mode=
"independent", cross_attn_rope=True, fuse_encoder_levels=True,
code_embed_mode="pq_table", fuse_position="post", fuse_use_null_kv=False,
steps=4000.

    uv run python -m qcute.qcute_refine_v4 --config configs/qcute_refine_v4_k32_narrow_postfuse_nonull_uncond.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_k32_narrow_postfuse_nonull_uncond
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
fuse_position = "post"
fuse_use_null_kv = False

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
