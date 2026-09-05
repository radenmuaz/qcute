"""image_gen_cifar base config -- minimal (d_model=256, n_layers=1) run over full CIFAR-10.

uv run python3 -m image_gen_cifar.run_causalattn --config image_gen_cifar/configs/base.py
"""

run_name = "cifar_base"

d_model = 256
n_layers = 1
n_heads = 4
code_vocab = 16    # per-chunk PQ codebook width
pq_chunks = 4       # combinatorial capacity code_vocab**pq_chunks = 65536 per code
strides = (2, 4, 4)
code_extract_mode = "mean"
decoder_mode = "seq"
class_conditional = False  # first training run: unconditional

epochs = 100
batch_size = 8  # decoder's effective batch is batch_size*img_size (32 columns) -- keep this small on MPS
lr = 3e-4
log_every = 100
eval_every_epochs = 1
qual_gen_n = 4
