"""v5_stack_windowtest/ks21_stageA: same as ks21_control.py but level0's SELF track window is
capped to K0=2 (block-local -- also caps byte-level self-attention itself, since both are
coupled through the same window value, see CLAUDE.md's investigation notes), while level0's
level1 (coarser) track stays unbounded, and level1 (top level) is untouched. Tests: does losing
WIDE self-attention/self-code, alone, cost meaningfully vs. the dense control?

attn_window = ((-1, (2, -1)), -1)
  level0: (encode_window=-1 unbounded, decode_windows=(self=2, level1-track=-1 unbounded))
  level1: -1 (top level, unchanged, encode=decode=self=unbounded)

uv run python -m qcute.qcute_v5 --decoder_type stack --config configs/v5_stack_windowtest/ks21_stageA.py

# plot after training:
uv run python scripts/plot_run.py logs/v5_stack_windowtest_ks21_stageA
"""
from pathlib import Path

run_name = "v5_stack_windowtest_ks21_stageA"
decoder_type = "stack"
Ks = (2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = ((-1, (2, -1)), -1)
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
