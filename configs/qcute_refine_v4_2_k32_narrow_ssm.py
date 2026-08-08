"""qcute.qcute_refine_v4_2 config: CLONE of configs/qcute_refine_v4_2_
k32_narrow.py, ONE change: `code_head_mode="chain"` + `bit_head_class=
"ssm"` instead of the default `code_head_mode="independent"`.

Session rationale: v4.2 re-gained a `code_head_mode` flag this session
("later re enable flag to use bitpredict heads, default indp now") after
BitPredictHead was made unreachable dead code in v4.2's first cut. This
run tests whether the EXACT chain-rule factorization (BitPredictHeadSSM
— linear-decay recurrence over the dq bits, see qcute_refine_v4_2.py's
own docstring: "s_j = alpha*s_{j-1} + bit_embed_{j-1}", a decayed
cumulative sum, not a plain independent-per-bit head) closes any of the
"independent-bit BCE is only an upper bound on true bits-per-byte" gap
`qcute_refine_v4_2_k32_narrow.py`'s own docstring flags — since the
ntp_head is now SHARED across every level (byte included, v4.2's whole
point), this also tests the chain head's own weight-sharing behavior
for the first time. `ssm` chosen over `attn`/`conv` for this first test:
cheapest of the three chain variants (linear recurrence, no attention or
conv overhead), and its own `_forward_fixed`/`h_scale`/decay-grid
tensors were just made more efficient this same session (precomputed
buffers instead of rebuilt every forward call).

Everything else identical to qcute_refine_v4_2_k32_narrow.py: Ks=(32,32),
dq=8, d_model=256, n_layers=1, context_len=1024, attn_window=(32,32),
fuse_encoder_levels=True, fuse_use_null_kv=False, steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_ssm.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_ssm
"""
from pathlib import Path

Ks = (32, 32)
dq = 8
d_model = 256
n_layers = 1
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (32, 32)

fuse_encoder_levels = True
fuse_use_null_kv = False
code_head_mode = "chain"
bit_head_class = "ssm"

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
