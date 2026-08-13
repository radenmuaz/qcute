"""qcute.qcute_refine_v4_2 DIAGNOSTIC config: `qcute_refine_v4_2_diag1`
-- strips the "narrow simplex" family (qcute_refine_v4_2_k32_narrow_
simplex_l2*) down to LEVEL 0 ONLY, unconditional, no hierarchy, no
fusion at all: `Ks=(32,)` (n_levels=1 -- nothing above level 0 for it
to ever fuse with; `fuse_encoder_levels=False` is moot but set for
clarity), `context_len=32` (matches configs/bytelm_xs1_ctx32.py's own
`context=32` EXACTLY, for a direct apples-to-apples comparison),
`n_layers=1`/`d_model=256`/`n_heads=4` (matches bytelm's "xs" preset
exactly too), `attn_window=(-1,)` (dense -- at context_len=32 a window
of 32 is dense anyway, set explicitly for clarity, matching bytelm's
own always-dense attention).

Session: "you are wrong, even bytelm 1 layer small context 32 is good
[on train set]" -- bytelm_xs1_ctx32 (1 layer, context=32, trained on
the FULL 900K corpus, train_bpb=2.66, never memorized) generates
coherent English word fragments on train prompts
(scripts/qual_gen_bytelm.py's own output: "requently Asked", "n the
alchemists"). qcute_refine_v4_2's own "narrow simplex" family, even
when it memorizes its (tiny) training data almost perfectly
(train_bpb=0.14-0.21, byte_acc=97%), generates GARBAGE on those exact
same train prompts. That comparison rules out exposure bias as the
sole explanation (higher teacher-forced accuracy should mean MORE
robustness to compounding errors, not less) and points at something
specific to `qcute_refine`'s own machinery.

This config isolates WHERE: strip away every hierarchical/fusion
addition (the thing `qcute_refine` adds ON TOP of a plain causal
transformer) and test whether the bare LevelLM trunk alone -- same
CausalSelfAttention/Block/CodeEmbed classes this whole file already
uses, just with n_levels=1 so no fusion/code-handoff machinery ever
runs -- can match bytelm_xs1_ctx32's generation quality at the SAME
context length and layer count. If it CAN: the problem is specifically
in the hierarchy/fusion layer, not this file's own base transformer
implementation. If it CAN'T: the problem is more fundamental, in
LevelLM's own trunk/embedding mechanics, independent of hierarchy.

`quant_type="simplex"` kept (not switched to "bsq") deliberately --
matches the "narrow simplex" family's own level-0 embedding/mechanics
exactly, isolating "remove the hierarchy" as the ONLY variable changed
relative to that family, not also changing level 0's own
representation. `code_bits`/`simplex_untie_head`/`untie_levels` are all
moot at n_levels=1 (nothing above level 0 to share or untie with) --
left at their defaults.

`qual_gen_bytes=16`/`qual_prompt_bytes=16`/`qual_source="train"` (new
this session, qcute_refine_v4_2.py's train()) -- matches
scripts/qual_gen_bytelm.py's own prompt_bytes=16/gen_bytes=16 used for
the bytelm_xs1_ctx32 comparison run, both fitting inside
context_len=32, live every eval round.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_diag1.py

    # watch live:
    tail -f logs/qcute_refine_v4_2_diag1/run.log
"""
from pathlib import Path

Ks = (32,)
d_model = 256
n_layers = 1
context_len = 32

n_heads = 4
mlp_mult = 4
attn_window = (-1,)

fuse_encoder_levels = False
quant_type = "simplex"

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

qual_gen_bytes = 16
qual_source = "train"
qual_prompt_bytes = 16
