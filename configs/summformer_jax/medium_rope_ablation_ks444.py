"""summformer_jax.train_summformer config: second medium ablation variant, deeper Ks (Ks=(4,4,4),
n_fuse=3) with n_layers=3 -- effective_depth = 3*(1+2*3) = 21, the max n_layers that keeps
effective_depth <= gpt2-medium's own n_layer=24 (n_layers=4 would give eff_depth=28, over the cap).
Chosen over n_layers=2 (eff_depth=14, params -21.0%, but FLOPs only 0.602x baseline) specifically
to get FLOPs closer to the baseline (0.839x) at the cost of landing slightly ABOVE baseline params
(+3.9%, 367.6M vs gpt2-medium's 353.8M) rather than below -- an explicit tradeoff pick, see
docs/status_tpu.md's "second summformer ablation variants" note for the full sweep.

d_model=1024/n_heads=16 match GPT2-medium exactly, same as the original medium_rope_ablation.py --
this is a Ks/n_layers variant of that same comparison, not a different width.

    uv run python summformer_jax/train_summformer.py --config configs/summformer_jax/medium_rope_ablation_ks444.py
"""
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
Ks = "4,4,4"
d_model = 1024
n_heads = 16
n_layers = 3
sequence_length = 1024
batch_size = 4
total_batch_size = 524288
