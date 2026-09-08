"""image_lagcodec/run_lagcodec.py -- 1:1 port of qcute_lagcodec_decoder.py's StackDecoder track0
mechanism (encode_like_self_attn_decode + seed_query_decode, shared self/cross-attn weights, two-
pass K/V reuse -- see StackDecoderTrack0/Track0Layer docstrings). Track0 ONLY (cfg.cond_depth=1 in
the reference's terms: level0's bytes conditioned on codes[0]/level1's code only, no track1/level2
conditioning yet) -- codes above level0 are NOT decoded by a separate stage at all (that cascade-
of-decoders design in the OTHER decoder_type options was this file's own invention, not part of
the reference); they're only ever produced by HierEncoder's own per-level NTP heads, same as
every decoder_type here.

Generation/reconstruction not yet implemented for this decoder_type -- this run is a pure
overfit/teacher-forced-accuracy check (does the CORRECT track0 mechanism even learn to reconstruct
under teacher forcing, matching the reference's actual design, rather than the divergent seed-
prepended-as-token design the other decoder_types used).

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_stackdecoder_track0.py
"""

run_name = "cifar_lagcodec_overfit1000_stackdecoder_track0"

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
decoder_type = "stack_track0"
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
eval_every_epochs = 500
qual_gen_n = 8
qual_gen_greedy = True
qual_gen_temperature = 1.0
