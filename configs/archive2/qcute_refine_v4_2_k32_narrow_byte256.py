"""qcute.qcute_refine_v4_2 config: CLONE of configs/qcute_refine_v4_2_
k32_narrow.py, ONE change: `byte_head_256way=True` instead of the
default False.

Session rationale: `qcute_refine_v4_2_k32_narrow`'s own result (best
val_bpb 4.0369, docs/status.md) showed a genuine, persistent training
instability traced to the code level's own loss (val_level1_bpb_pass1
never stabilizes, std 0.522 even in the second half of training — see
docs/kv_contribution.md §11) — the SHARED head/embed forces byte-level
(256-way discrimination) and code-level (very different BSQ-code target
statistics) predictions through one set of weights, and the two tasks
appear to actively fight each other rather than settle into a shared
solution. This config tests the most direct ablation of that hypothesis
(session: "another hack is make ablation use regular byte 256-way head,
put as flag"): give level 0 its OWN, UNSHARED, exact 256-way softmax
head (`Config.byte_head_256way=True` — exactly v4's own byte_repr=
"embed" mode, matching bytelm.py's convention) while code levels (here,
just level 1, since n_levels=2) keep their own separate dq-bit head, and
the TRUNK (self.blocks/ln_f) stays shared across both levels regardless
(v4.2's core mechanism, unaffected by this flag). If this closes most of
the gap to the healthy ~2.5-2.6 K=32 range, that confirms it's
specifically forcing an EXACT-vs-approximate, differently-shaped
prediction task through the SAME weights that caused the instability —
not weight-sharing (via the trunk) in general, which `qcute_refine_v4_1_
k32_narrow_shared` (queued) tests in isolation from head-sharing
entirely.

Everything else identical to qcute_refine_v4_2_k32_narrow.py: Ks=(32,32),
dq=8, d_model=256, n_layers=1, context_len=1024, attn_window=(32,32),
fuse_encoder_levels=True, fuse_use_null_kv=False, code_head_mode=
"independent" (only affects the code levels' own now-separate head),
steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_byte256.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_byte256
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
byte_head_256way = True

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
