"""gpt2-medium seqlen ablation, matched to summformer_jax/lm/configs/sweep_seqlen_small_1's
total_batch_size=524288 and batch_size=8 (grad_accum derived the same way on both sides:
grad_accum_steps = total_batch_size // (batch_size * sequence_length * n_devices)) -- same
matched-tokens-per-step invariant discussed for the small/medium baselines. use_remat=True
(gradient checkpointing, see model_gpt.py's ModelConfig.use_remat) needed at these context
lengths, same as the summformer side.

    uv run python gpt2_jax/train_gpt.py --config gpt2_jax/configs/sweep_seqlen_medium_1/medium_seqlen2048.py
"""
model = "medium"
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
use_flash_attention = True
use_remat = True
batch_size = 8
sequence_length = 2048
total_batch_size = 524288
