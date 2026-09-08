"""image_lagcodec/run_lagcodec.py -- StageSelfAttnDecoder is structurally faithful to
qcute_lagcodec_decoder.py's StackDecoder (own-code-prepended-as-block-start-token self-attention,
K+1 slots per block, unshifted own-block target alignment -- confirmed by direct comparison, see
session notes). This config makes RECONSTRUCTION faithful too: sequential_decode=True forces
reconstruct_tree_sequential (ONE continuous causal chain, KV-cached but verified byte-identical to
a from-scratch zero-cache full-recompute implementation matching the reference's own
_stack_generate_blockwise exactly) -- NO recon_ncode block-local grouping/parallelization at all,
which is what the reference actually does by default (window=None, "sync... one continuous causal
chain across every block"; their windowed/grouped variant is explicitly labeled an "async
ablation", never the default).

start_level=0: test level0 decode in isolation first (given its own real code) -- the reference's
own "roundtrip" framing (check_roundtrip_consistency), not yet the full level2->level1->level0
chain. Longer training (6000 epochs) than prior sinkgd runs (2000) since surviving full-sequence
(1024-block) compounding needs teacher-forced accuracy pushed much higher than the ~99.7% ceiling
seen so far -- this is the actual overfit test: can genuine training (not a decode-mechanism
change) get reconstruction to work.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_stackdecoder_faithful.py
"""

run_name = "cifar_lagcodec_overfit1000_stackdecoder_faithful"

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
# recon_ncode irrelevant here -- sequential_decode=True bypasses it entirely
# dec_d_model/dec_n_layers/dec_n_heads/dec_n_kv_heads left None -> mirror d_model/n_layers/n_heads/n_kv_heads above

# --- training ---
batch_size = 16
n_devices = None
epochs = 6000
lr = 1e-2
warmup_steps = 100
weight_decay = 1e-5
optimizer = "sinkgd"
optimizer_kwargs = {}
seed = 0
train_subset_n = 100

# --- logging / eval ---
log_every = 500
eval_every_epochs = 1000
qual_gen_n = 8
qual_gen_greedy = True
qual_gen_temperature = 1.0
sequential_decode = True
start_level = 0
