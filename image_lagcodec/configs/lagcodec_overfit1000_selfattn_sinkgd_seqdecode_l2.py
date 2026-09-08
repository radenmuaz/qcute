"""image_lagcodec/run_lagcodec.py -- OVERFIT SANITY CHECK, decoder_type="self_attn", optimizer="sinkgd".
Reconstruction uses reconstruct_tree_sequential(start_level=2): stage decode level2->level1 as ONE
continuous causal chain covering ALL of level1's blocks (all of level2's real codes as context,
no recon_ncode grouping/isolation), then chain that result into level1->level0 the same way (one
continuous chain, all of level0's blocks). Like the reference qcute_lagcodec StackDecoder's cross-
attn-to-own-code mechanism, but with the code PREPENDED into the sequence (StageSelfAttnDecoder's
actual mechanism) instead of a separate cross-attention stack -- and staged across levels (2->1,
then 1->0), not just level0 alone. Training identical to the other ncode ablation configs
(sequential_decode/start_level only affect the periodic qualitative reconstruction step).

Point of comparison: lagcodec_overfit1000_selfattn_sinkgd_ncode_full.py reaches the same "no
group-boundary resets" property via recon_ncode maximized to each level's full block count
(a hack reusing reconstruct_group with g=n_blocks) but only chains from level0. This run uses the
dedicated reconstruct_tree_sequential method AND starts the chain at level2 (using level2's own
codes as the real entry point, exercising the multi-level chain fully) -- both should be
mechanistically equivalent at level0 if start_level were 0 for both; the level2 entry point is
the deliberate difference under test here.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_selfattn_sinkgd_seqdecode_l2.py
"""

run_name = "cifar_lagcodec_overfit1000_selfattn_sinkgd_seqdecode_l2"

# --- model ---
img_size = 32
d_model = (256, 256, 256)
n_layers = (2, 2, 2)
n_heads = (4, 4, 4)
n_kv_heads = (None, None, None)
strides = (3, 4, 4)
code_vocab = 8
pq_chunks = 4
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
decoder_type = "self_attn"
# dec_d_model/dec_n_layers/dec_n_heads/dec_n_kv_heads left None -> mirror d_model/n_layers/n_heads/n_kv_heads above

# --- training ---
batch_size = 16
n_devices = None
epochs = 2000
lr = 1e-2
warmup_steps = 100
weight_decay = 1e-5
optimizer = "sinkgd"
optimizer_kwargs = {}
seed = 0
train_subset_n = 100

# --- logging / eval ---
log_every = 200
eval_every_epochs = 1000
qual_gen_n = 8
qual_gen_greedy = True
qual_gen_temperature = 1.0
sequential_decode = True
start_level = 2
