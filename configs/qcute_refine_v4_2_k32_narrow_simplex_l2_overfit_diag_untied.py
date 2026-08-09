"""qcute.qcute_refine_v4_2 DIAGNOSTIC config: CLONE of configs/
qcute_refine_v4_2_k32_narrow_simplex_l2_overfit_diag.py, TWO changes:
`simplex_untie_head=True` AND `untie_levels=True`.

Session: the tiny-corpus overfit diagnostic's own run.log ("from run.log
seems not good") shows the SAME pathology as the full-corpus run even
under trivially-easy memorization conditions (n_bytes=16384,
~4000 epoch-equivalents of exposure) -- by step 799/4000 (byte_acc=58.5%,
val_bpb=3.74 and falling), qual_generated is still collapsing into tight
repetition loops ("in in in in in", "ororototorororor...",
"aseatatat as as..."), not converging toward the ground-truth sample.
Session: "clone config ...overfit_diag and untie weights", then "by
untie, make it like v4, different head, embed, lm transformer each
level" -- the FIRST untied variant only separated the NTP classifier
from the embedding WITHIN a level (`simplex_untie_head`); v4.2's own
defining feature (one shared trunk/embed/head across EVERY level) was
still fully active underneath it. `untie_levels=True` (new this session)
additionally reverts to v4's original per-level separation -- every
level builds a completely fresh trunk and embed/head/code_pre pool of
its own, no aliasing across levels at all. Together, this is the
fullest "like v4" separation this file can express, tested in the SAME
easy-overfit regime as the tied run -- isolates "is sharing itself (at
any granularity) part of what's suppressing generation quality" from
"is the corpus/step budget the bottleneck" as a THIRD, orthogonal cell
alongside the tied-tiny-corpus and untied-full-corpus runs already
queued.

Everything else identical to qcute_refine_v4_2_k32_narrow_simplex_l2_
overfit_diag.py: Ks=(32,32), d_model=256, n_layers=2, context_len=1024,
attn_window=(32,32), fuse_encoder_levels=True, fuse_use_null_kv=False,
quant_type="simplex", n_bytes=16384, val_frac=0.1, steps=4000,
qual_gen_bytes=64/qual_source="train"/qual_prompt_bytes=64 (live
in-training qualitative generation every eval round).

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_simplex_l2_overfit_diag_untied.py

    # watch live:
    tail -f logs/qcute_refine_v4_2_k32_narrow_simplex_l2_overfit_diag_untied/run.log
"""
from pathlib import Path

Ks = (32, 32)
d_model = 256
n_layers = 2
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (32, 32)

fuse_encoder_levels = True
fuse_use_null_kv = False
quant_type = "simplex"
simplex_untie_head = True
untie_levels = True

data = Path("datasets/enwik8_1M.gz")
n_bytes = 16384
val_frac = 0.1

steps = 4000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 50
eval_every = 100
eval_batches = 20

qual_gen_bytes = 64
qual_source = "train"
qual_prompt_bytes = 64
