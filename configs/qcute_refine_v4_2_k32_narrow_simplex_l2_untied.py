"""qcute.qcute_refine_v4_2 config: CLONE of configs/qcute_refine_v4_2_
k32_narrow_simplex_l2.py, TWO changes: `simplex_untie_head=True` AND
`untie_levels=True`.

Session: "easier if you can make 4.2 untie weight mode" -- classic
weight-tying ablation (Press & Wolf 2017), applied to the ONE place
`quant_type="simplex"` ties weights: the NTP classifier
(`F.linear(h, self.embed.weight)`) shares its weight matrix with the
INPUT embedding table by default. `simplex_untie_head=True` gives each
level its own PRIVATE `nn.Linear(D, V)` classifier instead
(`LevelLM.simplex_head`). Session follow-up, catching that this alone
wasn't the full ask: "by untie, make it like v4, different head, embed,
lm transformer each level" -- `simplex_untie_head` only untied the
classifier from the embedding WITHIN a level; v4.2's OWN defining
feature (one shared trunk/embed/head across EVERY level, byte included)
was still fully in effect underneath it. `untie_levels=True` (new this
session) reverts THAT to v4's original per-level separation: every
level builds a completely fresh self.blocks/ln_f/fuse_* trunk and a
fresh embed/head/code_pre pool of its own, no aliasing across levels at
all, for any quant_type. The two flags are orthogonal and composable --
this config sets both for the fullest "like v4" separation: distinct
trunk, distinct embed, AND distinct (untied) classifier per level. See
both flags' own Config docstrings for the full rationale, and note
Config.untie_levels' own docstring on the one legitimate (not a bug)
side effect: the TOP level's own code_pre gets no gradient once
untied (nothing consumes its upward code), previously masked by weight-
sharing -- verified via validate_generation still passing regardless.

Direct comparison point against simplex_l2.py (best.pt: best_val_bpb=
2.5892, but qualitative generation on both train and val data -- even
with the LAST checkpoint on TRAINING data itself -- produced no
recognizable English, degenerating into repetitive fragments; see
scripts/qual_gen_v4_2.py's own output and the chat discussion) --
tests whether reducing the aggressiveness of weight sharing (the most
extreme point in this whole v4.2 family) recovers any of that lost
generation quality, independent of the tiny-corpus overfit diagnostic
(qcute_refine_v4_2_k32_narrow_simplex_l2_overfit_diag.py) running
concurrently in the queue.

Everything else identical to simplex_l2.py: Ks=(32,32), d_model=256,
n_layers=2, context_len=1024, attn_window=(32,32),
fuse_encoder_levels=True, fuse_use_null_kv=False, quant_type="simplex",
code_bits=8 (default), steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_simplex_l2_untied.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_simplex_l2_untied
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
