"""qcute.qcute_refine_v4_2 config: CLONE of configs/qcute_refine_v4_2_
k32_narrow_byte_softmax_head_only.py, ONE change: `byte_head_factored=
True` instead of `byte_softmax_head_only=True`.

Session ask: "use structured matrix but to replace dense linear map to
2**n way output softmax... some loss in repr ok for params saving." Same
narrow ablation shape as `byte_softmax_head_only` (level 0's INPUT
embedding and `code_pre` stay the SHARED dq-bit pool; only the OUTPUT
readout is private) — but the private readout is `FactoredSoftmaxHead`
(new this session): `logits[i,j] = f1(h)[i] + f2(h)[j]`, an outer sum of
two small `D->16` projections (`16*16=256=vocab`) instead of one dense
`D->256` matrix. Params: 8,224 vs. `byte_softmax_head_only`'s dense head
(65,792) — an 8x reduction in JUST this head; FLOPs: 16,384 vs. 131,072
(~8x). Still a genuine softmax over a real (if structured) logit vector —
no chain-rule/teacher-forcing needed, verified via fixed/loop-consistency-
style checks (full-model forward+backward clean, `validate_generation`
exact `generate_no_cache` vs `generate_kv_cache` match).

Representational cost (see FactoredSoftmaxHead's own docstring): the
256-way logit vector, reshaped [16,16], is constrained to row-effect +
column-effect (additively separable) — cannot express genuine "outcome i
needs outcome j specifically" interactions a dense head could. Queued
alongside `qcute_refine_v4_2_k32_narrow_byte_lowrank.py` (rank=16, same
param/FLOP budget) as the direct comparison point — see that config's own
docstring for why low-rank is expected to be STRICTLY more expressive at
matched budget (factored is a zero-free-parameter special case of
low-rank at rank=v1+v2).

Everything else identical to qcute_refine_v4_2_k32_narrow_byte_softmax_
head_only.py: Ks=(32,32), dq=8, d_model=256, n_layers=1, context_len=1024,
attn_window=(32,32), fuse_encoder_levels=True, fuse_use_null_kv=False,
code_head_mode="independent", steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_byte_factored.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_byte_factored
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
byte_head_factored = True

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
