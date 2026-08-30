"""End-to-end smoke test for the new canonical summformer.py wired into the image_classification
lineage: train step, eval (val accuracy), and classify-inference on a held-out example. Synthetic
byte-level "images" (random ints in [0,255), small size for CPU/TPU speed) so this runs in seconds
-- NOT a real training run, just a correctness check of the wiring (forward/backward/eval/classify
all execute and produce sane shapes/behavior) for the ClassifierHead + Encoder/Decoder chain.

    uv run python summformer_jax/image_classification/smoke_test_new_arch.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # summformer_jax/

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from summformer import SummFormer, Embedder, Encoder, Decoder, StackConfig, ChainStageConfig, CrossAttnSpec, ClassifierHead, topk_accuracy, cross_entropy

VOCAB_SIZE = 256  # byte-level RGB
D, H = 64, 4
IMG = 16  # tiny synthetic "image" side length -> context_len = IMG*IMG*3
L = IMG * IMG * 3
NUM_CLASSES = 10


def build():
    emb_cfg = StackConfig(n_layers=2, d_model=D, n_heads=H, window=16, compute_dtype=jnp.float32)
    embedder = Embedder(emb_cfg, context_len=L, vocab_size=VOCAB_SIZE, rngs=nnx.Rngs(0))
    chain = (ChainStageConfig(stride=8, n_layers=1, window=16), ChainStageConfig(stride=4, n_layers=1, window=16))
    encoder = Encoder(StackConfig(n_layers=0, d_model=D, n_heads=H, compute_dtype=jnp.float32), chain, context_len=L, rngs=nnx.Rngs(1))
    cross = (CrossAttnSpec(dst=1, encoder_output=0, window=16), CrossAttnSpec(dst=3, encoder_output=1, window=32))
    decoder = Decoder(StackConfig(n_layers=4, d_model=D, n_heads=H, window=16, compute_dtype=jnp.float32), cross, context_len=L, rngs=nnx.Rngs(2))
    model = SummFormer(embedder, encoder, decoder)
    return ClassifierHead(model, num_classes=NUM_CLASSES, pooling="mean", rngs=nnx.Rngs(3))


def make_synthetic_dataset(key, n_examples):
    kx, ky = jax.random.split(key)
    x = jax.random.randint(kx, (n_examples, L), 0, VOCAB_SIZE, dtype=jnp.int32)
    y = jax.random.randint(ky, (n_examples,), 0, NUM_CLASSES, dtype=jnp.int32)
    return x, y


def main():
    print(f"context_len={L} num_classes={NUM_CLASSES}")
    head = build()
    graphdef, state = nnx.split(head)
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(state)

    key = jax.random.PRNGKey(0)
    kt, kv = jax.random.split(key)
    train_x, train_y = make_synthetic_dataset(kt, 32)
    val_x, val_y = make_synthetic_dataset(kv, 16)

    def loss_fn(state, x, y):
        m = nnx.merge(graphdef, state)
        logits = m(x)
        loss = cross_entropy_logits = -jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), y[:, None], axis=-1).mean()
        top1 = topk_accuracy(logits, y, 1)
        return loss, top1

    @jax.jit
    def train_step(state, opt_state, x, y):
        (loss, top1), grads = jax.value_and_grad(loss_fn, has_aux=True)(state, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)
        return state, opt_state, loss, top1

    print("=== training ===")
    batch_size = 8
    for step in range(20):
        i = (step * batch_size) % max(1, len(train_x) - batch_size)
        x, y = train_x[i:i + batch_size], train_y[i:i + batch_size]
        state, opt_state, loss, top1 = train_step(state, opt_state, x, y)
        if step % 5 == 0:
            print(f"step {step}: loss={float(loss):.4f} top1={float(top1):.4f}")

    print("=== eval (val accuracy) ===")
    m = nnx.merge(graphdef, state)
    val_logits = m(val_x)
    val_loss = -jnp.take_along_axis(jax.nn.log_softmax(val_logits, axis=-1), val_y[:, None], axis=-1).mean()
    val_top1 = topk_accuracy(val_logits, val_y, 1)
    val_top5 = topk_accuracy(val_logits, val_y, min(5, NUM_CLASSES))
    print(f"val_loss={float(val_loss):.4f} val_top1={float(val_top1):.4f} val_top5={float(val_top5):.4f}")
    assert jnp.isfinite(val_loss), "val loss is not finite -- eval path broken"

    print("=== inference: classify a single held-out example ===")
    m = nnx.merge(graphdef, state)
    example_x = val_x[0:1]
    example_y = int(val_y[0])
    logits = m(example_x)
    pred = int(jnp.argmax(logits, axis=-1)[0])
    probs = jax.nn.softmax(logits, axis=-1)[0]
    print(f"true_label={example_y} predicted_label={pred} logits_shape={logits.shape} top_prob={float(probs[pred]):.4f}")
    assert logits.shape == (1, NUM_CLASSES), f"unexpected logits shape {logits.shape}"

    print("=== bidirectional pooling variant sanity check ===")
    head_bidir = ClassifierHead(head.model, num_classes=NUM_CLASSES, pooling="bidirectional", rngs=nnx.Rngs(4))
    logits_bidir = head_bidir(example_x)
    print(f"bidirectional logits_shape={logits_bidir.shape}")
    assert logits_bidir.shape == (1, NUM_CLASSES)

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
