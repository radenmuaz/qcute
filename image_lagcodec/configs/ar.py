"""Full CIFAR-10 (chat 2026-09-12) -- first run this session NOT using the 100-image overfit-
sanity subset (train_subset_n=None -> the real 50,000-image training set). Curriculum ENABLED
(phase-by-phase, not --no_curriculum): epochs_per_phase=(20,20,20,100) -- phases 1-3 (levels
0,1,2 alone) get 20 epochs each (~62,500 steps at batch_size=16 on the full set), phase 4 (all 4
levels jointly, no_freeze) gets 100 epochs (~312,500 steps). traversal="zorder", byte_group=3, token_head_type="ar"
everywhere -- tiny causal chain per-position, token_dim sized per-level (128 for level0's large
byte alphabet, 32 for levels1-3's small PQ-code alphabet). No true MTP (mtp_horizon=1).

uv run python3 -m image_lagcodec.run_lagcodec_zorder --config image_lagcodec/configs/cifar10_full_zorder_ar.py
uv run python -m image_lagcodec.run_lagcodec_zorder --config image_lagcodec/configs/cifar10_full_zorder_ar.py 2>&1 | tee ~/cifar10_full_zorder_ar.log
"""
'''
rsync -avz --delete \
  -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  --exclude=".git/" --exclude="__pycache__/" --exclude="*.pyc" --exclude=".venv/" \
  --exclude=".pytest_cache/" --exclude=".ruff_cache/" --exclude=".mypy_cache/" --exclude="datasets/" \
  --exclude="logs/" --exclude="checkpoints/" --exclude="*/logs/" --exclude="*/checkpoints/" --exclude=".env" \
  /Users/muaz/code/qcute/ muaz@35.186.98.243:~/qcute/
'''

'''
rsync -avz --delete \
  -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  --filter=':- ../.gitignore' \
  --exclude=".git/" \
  /Users/muaz/code/qcute/image_lagcodec/ muaz@35.186.98.243:~/qcute/image_lagcodec/
'''
'''
rsync -avz --delete \
  -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  --filter=':- .gitignore' \
  --exclude=".git/" \
  /Users/muaz/code/qcute/ muaz@35.186.98.243:~/qcute/

'''

'''
rsync -avz \
  -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  --exclude="checkpoints/" \
  muaz@35.186.98.243:~/qcute/image_lagcodec/logs/cifar10_full_zorder_mtp_ar_ar/ \
  /Users/muaz/code/qcute/image_lagcodec/logs/cifar10_full_zorder_mtp_ar_ar/

'''

# --- model ---
img_size = 32
d_model = (512, 512, 512, 512,)
n_layers = (2, 2, 2, 2,)
n_heads = (4, 4, 4, 4,)
n_kv_heads = (None, None, None, None,)
strides = (4, 4, 4, -1)
code_vocab = (16, 16, 16, 16)
pq_chunks = (4, 4, 4, 4)
mlp_mult = 2
rope_base = 10000.0
ntp_weight = 1.0
decoder_ncodes = 4
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "argmax"
cascade_rollout_prob = 0.5
init_scheme = "zero"
# init_scheme = "llama"
use_xsa = True
use_qknorm = True

byte_group = 3
token_head_type = "ar"
token_dim = (128, 32, 32, 32,)
token_n_heads = 2
mtp_horizon = 1
mtp_mode = "parallel"
traversal = "zorder"

# --- training ---
batch_size = 256
# epochs_per_phase = (20, 20, 100, )
epochs_per_phase = (200, 200, 200, )
warmup_steps = 100
grad_clip = 10.0
seed = 0
train_subset_n = None

# lr = 1e-3
# weight_decay = 1e-5
# optimizer = "adamw"
# optimizer_kwargs = {}

lr = 1e-1
# lr = 1e-2
weight_decay = 0
optimizer = "sinkgd"
optimizer_kwargs = {"sinkhorn_iters": 2, "weight_decay": 0}

# --- logging ---
log_every = 100
qual_gen_n = 8
