"""gpt2-medium seqlen ablation -- see medium_seqlen2048.py's docstring for the matching rationale.

    uv run python gpt2_jax/train_gpt.py --config gpt2_jax/configs/sweep_seqlen_medium_1/medium_seqlen8192.py
"""
model = "medium"
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
use_flash_attention = True
use_remat = True
batch_size = 8
sequence_length = 8192
total_batch_size = 524288
