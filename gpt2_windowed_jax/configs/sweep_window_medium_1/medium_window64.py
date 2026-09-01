"""gpt2_windowed_jax window-size ablation, medium model -- Cable's own "medium" shape (n_layer=24,
n_head=16, n_embd=1024, ~353.8M params), rope pos_method, context=1024, paper-faithful
total_batch_size=524288, batch_size=16/device (Cable's own MICRO_BATCH_SIZES["medium"]), same as
gpt2_jax/configs/medium_rope_default.py -- only `window` (bounded local self-attention, see
model_gpt.py's ModelConfig.window) varies across this sweep (64/128/256/512), isolating window
size as the single variable against the dense gpt2-medium baseline (`medium_paper_match`).
use_flash_attention left off -- window>0 bypasses the flash-attention kernel path entirely (see
CausalSelfAttention.__call__), setting it True here would be misleading.

    uv run python gpt2_windowed_jax/train_gpt.py --config gpt2_windowed_jax/configs/sweep_window_medium_1/medium_window64.py
"""
model = "medium"
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
use_flash_attention = False
window = 64
batch_size = 16
total_batch_size = 524288
