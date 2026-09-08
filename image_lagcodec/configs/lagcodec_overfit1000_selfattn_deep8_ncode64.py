"""image_lagcodec/run_lagcodec.py -- OVERFIT SANITY CHECK, decoder_type="self_attn" (original,
DIVERGENT-from-reference StageSelfAttnDecoder -- see stack_track0 for the faithful port), deep
8-level hierarchy strides=(2,)*8 (vs the usual 3-level (3,4,4)) -- tests whether a deeper/narrower
hierarchy helps this architecture overfit better. n_units per level (SEQ_LEN=3072):
L0=1536,L1=768,L2=384,L3=192,L4=96,L5=48,L6=24,L7=12.

recon_ncode=(64,)*8 -- one of a 4-way ablation (64/128/256/512 on tpu1-4), divisors of L0=1536.
Training unaffected by this (teacher-forced full-sequence always); only the periodic
reconstruction snapshot's group size differs.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_selfattn_deep8_ncode64.py
"""

run_name = "cifar_lagcodec_overfit1000_selfattn_deep8_ncode64"

# --- model ---
img_size = 32
d_model = (256,) * 8
n_layers = (2,) * 8
n_heads = (4,) * 8
n_kv_heads = (None,) * 8
strides = (2,) * 8
code_vocab = 8
pq_chunks = 4
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
decoder_type = "self_attn"
recon_ncode = (64,) * 8
# dec_d_model/dec_n_layers/dec_n_heads/dec_n_kv_heads left None -> mirror d_model/n_layers/n_heads/n_kv_heads above

# --- training ---
batch_size = 16
n_devices = None
epochs = 3000
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
start_level = 0
