"""v5_stack_windowtest/ks21_stageB: same as ks21_stageA.py but ALSO caps level0's level1
(coarser) track window to 4 (= 2 level1 codes back, cum_K=2 bytes/code for Ks=(2,1) since
Ks[1]=1 adds no further compression). This is the direct test of the actual parallel-decode
precondition: does decode only need a SMALL window into the level above, or does it silently
rely on long-range coarser context that this ablation would expose as a bpb regression?

attn_window = ((-1, (2, 4)), -1)
  level0: (encode_window=-1 unbounded, decode_windows=(self=2, level1-track=4))
  level1: -1 (top level, unchanged)

uv run python -m qcute.qcute_v5 --decoder_type stack --config configs/v5_stack_windowtest/ks21_stageB.py

# plot after training:
uv run python scripts/plot_run.py logs/v5_stack_windowtest_ks21_stageB
"""
from pathlib import Path

run_name = "v5_stack_windowtest_ks21_stageB"
decoder_type = "stack"
Ks = (2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = ((-1, (2, 4)), -1)
code_hard = True
code_sample = False
quant_type = "grid"
grid_dq = 16
grid_levels = 8
vocab = 256
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

data = Path("datasets/enwik8_1M.gz")
n_bytes = 100000
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 200
eval_every = 2000
eval_batches = 20
full_val_eval = True

qual_gen_bytes = 128
qual_prompt_bytes = 64
