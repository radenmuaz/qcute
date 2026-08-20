"""v1_stack_simplex/ks1_v256_pq1: qcute_v1 baseline at n_levels=1 -- level0 IS the top level here,
so decode is the UNCHANGED-from-v5 genuine self-code-recurrent NTP path (StackDecoder's is_top
branch), not the new BOS-interleaved/cross-attend-own-code mechanism (that only applies to
non-top levels, which don't exist when Ks=(1,)). Exists as: (1) a same-scale bpb reference point
against ks21_v256_pq1.py's n_levels=2 run, and (2) the "truly autoencode, cannot extrapolate"
degenerate case from docs/qcute_v1_plan.md's generation-feasibility section -- real generation at
n_levels=1 can only ever go through path (a) (draft via uncond LM -> encode -> decode-refine),
never path (b) (no level above to predict a code from). quant_type=simplex, vocab=256, pq_chunks=1
(paired against ks1_v64_pq4.py). Follows CLAUDE.md's standing overfit10k methodology
(n_bytes=10000, context=256, steps=1000).

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks1_v256_pq1.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks1_v256_pq1
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks1_v256_pq1"
decoder_type = "stack"
Ks = (1,)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 256
pq_chunks = 1
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 0
