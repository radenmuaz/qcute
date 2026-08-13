"""qcute.qcute_refine_v4_2 config: CLONE of configs/qcute_refine_v4_2_
k32_narrow_simplex_l2_postcross.py, THREE changes: `untie_levels=True`,
`untie_fusion_pass=True`, `simplex_untie_head=True`.

Session: "also define another config make the each level-pass separate
weights, as if there is 3 level lm, first 2 uncond, the level 0 pass 2
is cross attn only level... second isolate by allow free weight
different pass... by free weight this means like a fresh level lm own
embed, transformer, linear head."

Combined, this config trains what is effectively THREE independent
LMs sharing nothing:
  1. level 0's own unconditional pass (PASS 1) -- own embed, own
     self.blocks/ln_f trunk, own classifier (`untie_levels=True` gives
     level 0 its own fresh everything, same as
     qcute_refine_v4_2_k32_narrow_simplex_l2_untied.py).
  2. level 1's own unconditional pass (PASS 1, the top level, never
     fuses regardless) -- own embed/trunk/classifier too, via the same
     `untie_levels=True`.
  3. level 0's own FUSED pass (PASS 2) -- `untie_fusion_pass=True` gives
     this its own `embed_pass2`/`blocks_pass2`/`ln_f_pass2`, entirely
     separate from (1)'s own identity; `simplex_untie_head=True` also
     gives it (and (1)/(2), via the orthogonal per-level classifier-
     untie) its own private classifier rather than a weight-tied one --
     "own embed, transformer, linear head," the full "fresh level lm"
     ask. This pass ADDITIONALLY reads level 1's own hidden state via
     the genuine cross-attention fusion mechanism
     (`fuse_mode="cross_attn_post"`, same as the non-free config) --
     the ONE thing that makes it not simply a 4th independent
     unconditional model.

Direct comparison point against
qcute_refine_v4_2_k32_narrow_simplex_l2_postcross.py (default sharing,
same fuse_mode) -- isolates whether v4.2's own extreme, unconditional
weight-SHARING (not just the concat-vs-cross-attention fusion
mechanism) is itself part of what suppresses generation quality, same
question qcute_refine_v4_2_k32_narrow_simplex_l2_untied.py already
asked for concat fusion, now asked for real cross-attention fusion.

Everything else identical to the postcross config: Ks=(32,32),
d_model=256, n_layers=2, context_len=1024, attn_window=(32,32),
fuse_encoder_levels=True, fuse_use_null_kv=False, quant_type="simplex",
fuse_mode="cross_attn_post", code_bits=8 (default), steps=4000,
qual_gen_bytes=64/qual_prompt_bytes=64 (both train AND val prompts,
both uncond AND pass-2 generation, every eval round).

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_simplex_l2_postcross_free.py

    # watch live:
    tail -f logs/qcute_refine_v4_2_k32_narrow_simplex_l2_postcross_free/run.log
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
fuse_mode = "cross_attn_post"
untie_levels = True
untie_fusion_pass = True
simplex_untie_head = True

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
qual_prompt_bytes = 64
