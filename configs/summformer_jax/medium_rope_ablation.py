"""summformer_jax.train_summformer config: ablation of the Ks-hierarchical-summarization +
fuse-cross-attn method against configs/gpt2_jax/medium_rope_default.py's GPT2-medium baseline.
Same total_batch_size (524288, matching the paper's own value) and same grad_accum formula --
B=4, total_batch_size=524288, seq_len=1024, n_devices=4 -> grad_accum_steps=524288//(4*1024*4)=32
(a smaller per-device B than gpt2_jax's medium B=16 since this architecture's fuse-stage compute
is heavier per forward pass -- chosen conservatively to avoid OOM, not yet tuned upward; see
docs/status_tpu.md).

d_model=1024/n_heads=16 match GPT2-medium exactly; n_layers=2 + Ks=(2,2,2,2) (n_fuse=3) chosen so
this architecture's effective GPT2-equivalent depth (~n_layers*(1+2*n_fuse)=14) and param count
(~279M) land close to GPT2-medium's 24 layers / 353.8M params -- see the chat history around
2026-08-27 for the full FLOPs/param derivation and the -1-vs-2-Ks-length tradeoff analysis.

    uv run python summformer_jax/train_summformer.py --config configs/summformer_jax/medium_rope_ablation.py
"""
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
Ks = "2,2,2,2"
d_model = 1024
n_heads = 16
n_layers = 2
sequence_length = 1024
batch_size = 4
total_batch_size = 524288
