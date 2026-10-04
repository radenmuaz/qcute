"""
uv run python3 -m image_lagcodec.run_lagcodec_res_denoise --config image_lagcodec/configs/folder_audio_overfit.py
uv run python3 -m image_lagcodec.run_lagcodec_res_denoise_torch --config image_lagcodec/configs/folder_audio_overfit.py --device auto
"""
# Folder audio dataset: any soundfile-readable file (stdlib wave fallback for PCM wav), mixed to mono,
# band-limited resample to audio_sample_rate, 8-bit mu-law (pcm16: byte_group=2, pq_chunks=2), cut into
# seq_len-sample windows. Code labels = anti-aliased decimation of the waveform (audio_resample_label_fn).
dataset = "folder"
data_root = "datasets/audio"
modality = "audio"
audio_sample_rate = 16000
audio_encoding = "mulaw8"
seq_len = 4096
byte_group = 1
pq_chunks = 1
code_vocab = 256
traversal = "raster"
codelm_d_model = 256
codelm_n_layers = 2
codelm_n_heads = 2
codelm_n_kv_heads = 2
pq_dim = 64
mlp_mult = 4
ntp_weight = 1.0
label_reg_weight = 1.0
bos_rate_mode = "absolute"
strides = (4, 4)
attn_window = 1024
upsampler_ncodes = 1
downsampler_d_model = 128
downsampler_n_layers = 1
downsampler_n_heads = 1
downsampler_n_kv_heads = 1
downsampler_window = 1
upsampler_d_model = 512
upsampler_n_layers = 4
upsampler_n_heads = 2
upsampler_n_kv_heads = 2
upsampler_window = 2
share_across_levels = False
curriculum_mode = "freeze"
quantize_mode = "reinmax_limit"
level_gt_drop = 0.0
use_sink = True
precision = "bf16"
token_head_type = "ar"
codelm_token_head = "ar"
pardec_token_head = "ar"
token_dim = 64
token_n_heads = 2
eval_gen_train = True
gen_eval_all_levels = True
gen_eval_teacher_force_sanity = True
level_cycles = 2

# --- training ---
batch_size = 4
val_batch_size = 4
level_steps = (20_000, 20_000)
seed = 0
gen_eval_every_step = 5000
epoch_verbose = False
grad_clip = 1.0
lr = 5e-4
lr_schedule = "cosine"
lr_min = 1e-5
warmup_steps = 1000
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=0)
log_every = 500
ckpt_every_step = 10000
ckpt_keep = 1
