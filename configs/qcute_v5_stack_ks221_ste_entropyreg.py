"""qcute_v5_stack_ks221_ste_entropyreg: Ks=(2,2,1), context_len=256, attn_window=-1 (unbounded),
quant_type="softmax" (default), code_sample_mode="ste" -- deeper 3-level counterpart to
configs/qcute_v5_stack_ks41_ste.py (Ks=(4,1)), with Config.entropy_reg_weight=0.1 turned ON (that
run and its concat counterpart both left it at the 0.0/off default). softmax_entropy_reg's usage
diagnostic (scripts/measure_code_entropy.py) showed qcute_v5_stack_ks41_ste's softmax codebook
already reasonably utilized (55-59% of 256 ids at level0, 27-28% at level1) without any
regularization -- this run tests whether entropy_reg_weight>0 measurably improves that further (or
just costs bpb) on a harder/deeper 3-level config, not just confirms the diagnostic's own read.

uv run python -m qcute.qcute_v5_stack --config configs/qcute_v5_stack_ks221_ste_entropyreg.py

# plot after training:
uv run python scripts/plot_run.py logs/qcute_v5_stack_ks221_ste_entropyreg

# measure code usage after training (add this checkpoint name to CHECKPOINTS in the script first):
uv run python scripts/measure_code_entropy.py qcute_v5_stack_ks221_ste_entropyreg
"""
from pathlib import Path

Ks = (2, 2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_sample_mode = "ste"
entropy_reg_weight = 0.1

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 200
eval_every = 2000
eval_batches = 20

qual_gen_bytes = 128
qual_prompt_bytes = 64
