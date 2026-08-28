"""summformer_jax.train_summformer config: second small ablation variant, deeper Ks (Ks=(4,4),
n_fuse=2) with n_layers=2 -- effective_depth = 2*(1+2*2) = 10, the max n_layers that keeps
effective_depth <= gpt2-small's own n_layer=12 (n_layers=3 would give eff_depth=15, over the cap).
Chosen over n_layers=1 (eff_depth=5, params -8.9%, but FLOPs only 0.558x baseline) specifically to
get FLOPs closer to the baseline (0.846x) at the cost of landing slightly ABOVE baseline params
(+19.8%, 148.2M vs gpt2-small's 123.7M) rather than below -- an explicit tradeoff pick, see
docs/status_tpu.md's "second summformer ablation variants" note for the full sweep.

d_model=768/n_heads=12 match GPT2-small exactly, same as the original small_rope_ablation.py --
this is a Ks/n_layers variant of that same comparison, not a different width.

    uv run python summformer_jax/lm/train_summformer.py --config configs/summformer_jax/small_rope_ablation_ks44.py
"""
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
Ks = "4,4"
d_model = 768
n_heads = 12
n_layers = 2
sequence_length = 1024
batch_size = 4
total_batch_size = 524288
