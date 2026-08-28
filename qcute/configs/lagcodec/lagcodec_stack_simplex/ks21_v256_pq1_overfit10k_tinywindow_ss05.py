"""v1_stack_simplex/ks21_v256_pq1_overfit10k_tinywindow_ss05: sanity check for the
scheduled_sampling_p fallback (see ks221_v256_pq1_overfit10k_window4_relaxed_ss05.py for the real
test and full rationale) -- same Ks=(2,1) tiny-window handicap as
ks21_v256_pq1_overfit10k_tinywindow.py but with scheduled_sampling_p=0.5: with probability 0.5,
non-top-level decode is fed the level-above's own sampled prediction instead of the real code
during training, closing the train/inference distribution gap that's the suspected cause of the
ks221 real-generation collapse seen across every quant-type/cond_depth/window variant tried so far
(see docs/status.md's 2026-08-20/21 hard-convergence-queue entries). This is the last fallback in
the chain the user specified.

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks21_v256_pq1_overfit10k_tinywindow_ss05.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks21_v256_pq1_overfit10k_tinywindow_ss05
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks21_v256_pq1_overfit10k_tinywindow_ss05"
decoder_type = "stack"
Ks = (2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = ((-1, [2, -1]), -1)
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 256
pq_chunks = 1
scheduled_sampling_p = 0.5
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

qual_gen_bytes = 64
