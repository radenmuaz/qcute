"""gpt2_windowed_jax window-size ablation, small model -- Cable's own "small" shape (n_layer=12,
n_head=12, n_embd=768, ~124M params), rope pos_method, context=1024, paper-faithful
total_batch_size=524288, batch_size=32/device (Cable's own MICRO_BATCH_SIZES["small"]), same as
gpt2_jax/configs/small_rope_default.py -- only `window` (bounded local self-attention, see
model_gpt.py's ModelConfig.window) varies across this sweep (64/128/256/512), isolating window
size as the single variable against the dense gpt2-small baseline (`small_paper_match`).
use_flash_attention left off -- window>0 bypasses the flash-attention kernel path entirely (see
CausalSelfAttention.__call__), setting it True here would be misleading.

    uv run python gpt2_windowed_jax/train_gpt.py --config gpt2_windowed_jax/configs/sweep_window_small_1/small_window128.py
"""
model = "small"
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
use_flash_attention = False
window = 128
batch_size = 32
total_batch_size = 524288
