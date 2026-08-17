"""qcute_v5_stack_bsq16: BSQ quantization (quant_type="bsq", bsq_bits=16, the CodeEmbed
table-lookup ceiling -- MAX_PQ_TABLE_DQ in qcute/qcute_v5_stack.py) with stochastic bit
sampling enabled (code_sample_mode="sample" -- each bit drawn Bernoulli(sigmoid(v_unit))
instead of deterministic sign(), straight-through gradient unchanged), otherwise identical to
configs/qcute_v5_stack_3.py (Ks=(1,), context_len=256, attn_window=(256,)).

uv run python -m qcute.qcute_v5_stack --config configs/qcute_v5_stack_bsq16.py

# plot after training:
uv run python scripts/plot_run.py logs/qcute_v5_stack_bsq16
"""
from pathlib import Path

Ks = (1,)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (256,)
quant_type = "bsq"
bsq_bits = 16
code_sample_mode = "sample"

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 200
eval_every = 2000
# eval_every = 100
eval_batches = 20

qual_gen_bytes = 128
qual_prompt_bytes = 64
