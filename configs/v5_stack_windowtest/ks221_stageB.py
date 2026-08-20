"""v5_stack_windowtest/ks221_stageB: same as ks221_stageA.py but ALSO caps every coarser track's
window to ~2 codes back in that TRACK's own cum_K units: level0's level1-track and level2-track
both get window=8 (cum_K=4 bytes/code for both, since Ks[2]=1 adds no further byte-span beyond
level1 -- 2 codes * 4 = 8), level1's level2-track gets window=4 (cum_K=2 level0-code-units/code
-- 2 codes * 2 = 4). Direct test of the parallel-decode precondition at 3 levels: does decode
only need a SMALL window into the level(s) above, at every non-top level simultaneously?

attn_window = ((-1, (2, 8, 8)), (-1, (2, 4)), -1)
  level0: (encode=-1, decode=(self=2, level1-track=8, level2-track=8))
  level1: (encode=-1, decode=(self=2, level2-track=4))
  level2: -1 (top level, unchanged)

uv run python -m qcute.qcute_v5 --decoder_type stack --config configs/v5_stack_windowtest/ks221_stageB.py

# plot after training:
uv run python scripts/plot_run.py logs/v5_stack_windowtest_ks221_stageB
"""
from pathlib import Path

run_name = "v5_stack_windowtest_ks221_stageB"
decoder_type = "stack"
Ks = (2, 2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = ((-1, (2, 8, 8)), (-1, (2, 4)), -1)
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
