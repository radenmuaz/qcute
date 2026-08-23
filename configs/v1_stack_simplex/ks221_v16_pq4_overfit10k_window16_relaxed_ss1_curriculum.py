"""v1_stack_simplex/ks221_v16_pq4_overfit10k_window16_relaxed_ss1_curriculum: same setup as
ks221_v16_pq4_overfit10k_window16_relaxed_ss1.py (Ks=(2,2,1), window16_relaxed, vocab=16
pq_chunks=4, scheduled_sampling_p=1.0, detach_ss_sample=False, uncertainty_weighting=False) but
adds a level curriculum via the new active_srcs_mode/active_srcs_until_step Config fields
(qcute_v1_common.py): for step < active_srcs_until_step (here steps/2 = 1500), every decode_level call
uses max_srcs=2, which caps level0's decode to its nearest upper track only (level1's code) --
level2's cross-attn stage is skipped entirely, so the model trains like a ks21 submodel for the
first half. From step 1500 onward, max_srcs reverts to None (full ks221, all cross-attn stages).
Motivation: `uw_ss1`/`ss1` (both isolation-pair runs, see docs/status.md's hard-convergence-queue
entries) still collapsed to repetitive single-token generation even at 98%+ train byte_acc --
testing whether letting the simpler (byte, level1) dependency converge before introducing level2
conditioning avoids that collapse.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks221_v16_pq4_overfit10k_window16_relaxed_ss1_curriculum.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v16_pq4_overfit10k_window16_relaxed_ss1_curriculum
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v16_pq4_overfit10k_window16_relaxed_ss1_curriculum"
decoder_type = "stack"
Ks = (2, 2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (
    (-1, [32, -1, -1]),
    (-1, [32, -1]),
    -1,
)
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 16
pq_chunks = 4
scheduled_sampling_p = 1.0
detach_ss_sample = False
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

steps = 3000
active_srcs_mode = 2
active_srcs_until_step = steps // 2

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 64
