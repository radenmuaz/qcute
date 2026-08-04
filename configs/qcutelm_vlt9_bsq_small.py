"""qcute.qcutelm_vlt9 config: deliberately SMALL scale — context_len=128
(n_blocks=32), not qcutelm_vlt7/vlt8's usual 1024/256. Measured directly:
n_blocks=32 already takes ~2.1s/train-step on CPU (batch=4); the
block-by-block loop is roughly O(n_blocks^2) (each of n_blocks iterations
recomputes attention over the growing sequence), so n_blocks=256 (matching
the other qcutelm_vlt7/vlt8 runs) would be ~64x slower — days for an
8000-step run, not hours. This is the expected "prefill slow" cost of
genuine architectural symmetry (qcutelm_vlt9's whole point), not a bug —
see qcutelm_vlt9.py's module docstring. This config trades scale for
tractability: same K/dq/quant_type/d_model/lm_d_model as the other bsq
runs (architecture held constant), just a much shorter context and step
budget, enough to see whether true symmetry changes code_conditioned_acc/
within_block_acc/bpb trends at all — not meant to be bpb-competitive with
the full-scale qcutelm_vlt7/vlt8/baseline runs.

    uv run python -m qcute.qcutelm_vlt9 --config configs/qcutelm_vlt9_bsq_small.py
"""
from pathlib import Path

K = 4
context_len = 128   # n_blocks=32 — see module docstring for why not 1024
dq = 13
quant_type = "bsq"
fsq_levels = 8

d_model = 96
n_heads = 4
n_layers = 2
mlp_mult = 4

lm_d_model = 256
lm_n_heads = 4
lm_n_layers = 3
lm_mlp_mult = 4

code_match_weight = 1.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 2000       # reduced from the usual 8000 — see module docstring
batch_size = 8      # reduced from the usual 16 — loop cost scales with batch too
lr_peak = 6e-4
weight_decay = 1e-5
warmup_steps = 200
cosine_decay = False
constant_steps = 100
eval_every = 100
eval_batches = 10

gen_every = 500
gen_prompt_len = 32
gen_new_bytes = 32
