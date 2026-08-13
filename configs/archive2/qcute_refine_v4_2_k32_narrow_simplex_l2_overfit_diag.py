"""qcute.qcute_refine_v4_2 DIAGNOSTIC config: CLONE of configs/
qcute_refine_v4_2_k32_narrow_simplex_l2.py, but on a TINY dataset and
with live in-training qualitative generation, to answer one question:
can this architecture even overfit/memorize a corpus small enough that
memorization should be trivial? Session: "queue on diagnostic run use
simplex l2 config, try tiny samples 1000 try to overfit... goal is to
get generation close to train sample else conclude degenerate arch."

Prompted directly by scripts/qual_gen_v4_2.py's own finding: simplex_l2's
LAST checkpoint (step=4000, train_bpb=1.56), greedy-decoded on its OWN
TRAINING data, produced no recognizable English -- fragments like "trest",
"tereanatis", repeated "the the the" loops, not the near-verbatim recall
a model that actually fit train_bpb=1.56 should manage. That result is
consistent with EITHER "the full 900K-byte corpus is just too much to
memorize at this scale/step budget" (an unremarkable, expected outcome)
OR "the architecture itself can't cleanly overfit even a trivial amount
of data" (a real, structural problem, independent of dataset size). This
config isolates the two: shrink the corpus far enough that memorization
should be nearly free, and watch (live, via qual_gen_bytes below) whether
generation actually converges toward the training sample.

`n_bytes=16384` (NOT literally ~1000, per the session's own "tiny samples
1000" framing) -- RefineLM.__init__ asserts context_len must be an exact
multiple of Ks[0]*Ks[1]=1024 at this config's own Ks=(32,32)/context_len
=1024, and sample_context needs BOTH train and val regions strictly
LONGER than context_len=1024 to sample a window at all (else it errors
indexing past the end) -- an honest floor, not a stylistic choice.
`val_frac=0.1` on 16384 bytes gives train=14746/val=1638, both comfortably
above 1024. Still a ~98.2% reduction from the full ~900K-byte corpus:
1600x more passes over each byte at the same steps=4000/batch_size=16 as
every other config in this family (4000*16*1024/16384 ~= 4000
epoch-equivalents of exposure vs. ~4.5 for the full corpus) -- if
memorization doesn't show up here, it isn't a data-scale problem.

`qual_gen_bytes=64`/`qual_source="train"`/`qual_prompt_bytes=64` (new
this session, qcute_refine_v4_2.py's train()) -- AR-generates from a
TRAIN-region prompt EVERY eval round (eval_every=100 below, more frequent
than this family's usual 100 anyway, so ~40 qual-gen snapshots across the
run) and logs prompt/generated/ground_truth live, so run.log itself
becomes the record of whether generation is converging toward the sample
over training, not just a post-hoc check on the finished checkpoint.

Everything else identical to qcute_refine_v4_2_k32_narrow_simplex_l2.py:
Ks=(32,32), d_model=256, n_layers=2, context_len=1024, attn_window=
(32,32), fuse_encoder_levels=True, fuse_use_null_kv=False,
quant_type="simplex", code_bits=8 (default), steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_simplex_l2_overfit_diag.py

    # watch live:
    tail -f logs/qcute_refine_v4_2_k32_narrow_simplex_l2_overfit_diag/run.log
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
