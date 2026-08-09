"""qcute.qcute_refine_v4_2 config: CLONE of configs/qcute_refine_v4_2_
k32_narrow_simplex_l2.py, ONE change: `fuse_mode="cross_attn_post"`.

Session: "rerun with this: post cross attn, all ntp uncond and cond
(pass 2) on, every eval round, generate both train prompt and val
prompt, for level 0 uncond and level 0 pass 2 cross attn." Prompted by
confirming v4.2's own fusion mechanism is concat, not real
cross-attention (module docstring: the level-above's own hidden state
is appended to self.blocks' own K/V, no separate cross-attention
weights at all) -- this config reintroduces qcute_refine_v4.py's
ORIGINAL CrossBlock-based "post" fusion (see that class's own
docstring, ported near-verbatim into qcute_refine_v4_2.py this
session) as a genuine, separately-weighted cross-attention sublayer run
AFTER self.blocks/ln_f, instead of concat.

"all ntp uncond and cond (pass 2) on": unchanged from simplex_l2.py --
byte_ntp_weight/code_ntp_weight/fusion_ntp_weight all default to 1.0,
so level 0's own unconditional (PASS 1) loss, level 1's own
unconditional (PASS 1, the top level, never fuses) loss, AND level 0's
fused (PASS 2) loss are all simultaneously part of the training signal
-- nothing to change here, already the default.

"every eval round, generate both train prompt and val prompt... for
level 0 uncond and level 0 pass 2 cross attn": qcute_refine_v4_2.py's
own train()/qualitative_generate were extended this session to always
generate from BOTH train and val regions every eval round (no longer a
single qual_source choice), and for EACH prompt to show BOTH level 0's
unconditional (PASS-1-only, generate_level0_uncond -- ignores fusion
entirely) and level 0's PASS-2/fused (generate_no_cache, cross-attn
now instead of concat) continuations side by side -- a direct
within-model "does fusion actually help" comparison, not just against
an external baseline.

DEFAULT weight sharing (session: "first try default weight sharing") --
`untie_levels`/`untie_fusion_pass`/`simplex_untie_head` all left False:
PASS 1 and PASS 2 still share the SAME self.blocks/ln_f/embed (only the
fusion MECHANISM changed, concat -> cross-attention, not the sharing
scheme). Direct comparison point for
qcute_refine_v4_2_k32_narrow_simplex_l2_postcross_free.py, which
additionally frees every level-pass's own weights ("isolate by allow
free weight different pass").

Everything else identical to simplex_l2.py: Ks=(32,32), d_model=256,
n_layers=2, context_len=1024, attn_window=(32,32),
fuse_encoder_levels=True, fuse_use_null_kv=False, quant_type="simplex",
code_bits=8 (default), steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_simplex_l2_postcross.py

    # watch live:
    tail -f logs/qcute_refine_v4_2_k32_narrow_simplex_l2_postcross/run.log
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
