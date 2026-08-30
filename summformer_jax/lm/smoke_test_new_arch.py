"""End-to-end smoke test for the new canonical summformer.py wired into the lm (BPE) lineage:
train step, eval (val loss), and real prompt -> generate -> decode text, using tiktoken's GPT-2
BPE encoding (vocab_size=50257) -- confirms the new Encoder/Decoder/ARHead plumbing works for a
BPE vocab, not just byte-level. Tiny synthetic corpus (repeats of a fixed short text) so this runs
in seconds on CPU -- NOT a real training run, just a correctness check of the wiring
(forward/backward/eval/generate all execute and produce sane shapes/behavior).

    uv run python summformer_jax/lm/smoke_test_new_arch.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # summformer_jax/

import jax
import jax.numpy as jnp
import optax
import tiktoken
from flax import nnx

from summformer import SummFormer, Embedder, Encoder, Decoder, StackConfig, ChainStageConfig, CrossAttnSpec, ARHead

ENC = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = ENC.n_vocab

D, H, L = 64, 4, 128


def build():
    emb_cfg = StackConfig(n_layers=2, d_model=D, n_heads=H, window=16, compute_dtype=jnp.float32)
    embedder = Embedder(emb_cfg, context_len=L, vocab_size=VOCAB_SIZE, rngs=nnx.Rngs(0))
    chain = (ChainStageConfig(stride=8, n_layers=1, window=16), ChainStageConfig(stride=4, n_layers=1, window=16))
    encoder = Encoder(StackConfig(n_layers=0, d_model=D, n_heads=H, compute_dtype=jnp.float32), chain, context_len=L, rngs=nnx.Rngs(1))
    cross = (CrossAttnSpec(dst=1, encoder_output=0, window=16), CrossAttnSpec(dst=3, encoder_output=1, window=32))
    decoder = Decoder(StackConfig(n_layers=4, d_model=D, n_heads=H, window=16, compute_dtype=jnp.float32), cross, context_len=L, rngs=nnx.Rngs(2))
    model = SummFormer(embedder, encoder, decoder)
    return ARHead(model, vocab_size=VOCAB_SIZE, weight_tie=False, rngs=nnx.Rngs(3))


def make_synthetic_corpus(n_examples=32, seq_len=L):
    text = "The quick brown fox jumps over the lazy dog. " * 40
    ids = ENC.encode(text)
    data = []
    for i in range(n_examples):
        start = (i * 7) % max(1, len(ids) - seq_len - 1)
        data.append(ids[start:start + seq_len])
    return jnp.array(data, dtype=jnp.int32)


def main():
    print(f"vocab_size={VOCAB_SIZE}")
    head = build()
    graphdef, state = nnx.split(head)
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(state)

    train_data = make_synthetic_corpus(n_examples=32)
    val_data = make_synthetic_corpus(n_examples=8)

    def loss_fn(state, batch):
        m = nnx.merge(graphdef, state)
        loss, metrics = m(batch)
        return loss, metrics

    @jax.jit
    def train_step(state, opt_state, batch):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state, batch)
        updates, opt_state = optimizer.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)
        return state, opt_state, loss

    print("=== training ===")
    batch_size = 8
    for step in range(20):
        batch = train_data[(step * batch_size) % len(train_data): (step * batch_size) % len(train_data) + batch_size]
        if batch.shape[0] < batch_size:
            batch = train_data[:batch_size]
        state, opt_state, loss = train_step(state, opt_state, batch)
        if step % 5 == 0:
            print(f"step {step}: loss={float(loss):.4f}")

    print("=== eval (val loss) ===")
    m = nnx.merge(graphdef, state)
    val_loss, val_metrics = m(val_data)
    print(f"val_loss={float(val_loss):.4f} val_bpb={float(val_metrics['bpb']):.4f}")
    assert jnp.isfinite(val_loss), "val loss is not finite -- eval path broken"

    print("=== inference: generate + decode from a real prompt ===")
    m = nnx.merge(graphdef, state)
    prompt_text = "The quick brown"
    prompt_ids = jnp.array([ENC.encode(prompt_text)], dtype=jnp.int32)
    out_ids = m.generate_kv_cache(prompt_ids, n_new_tokens=10)
    decoded = ENC.decode(out_ids.tolist())
    print(f"prompt={prompt_text!r}")
    print(f"generated_ids={out_ids.tolist()}")
    print(f"decoded={decoded!r}")

    print("=== KV-cache consistency (same prompt) ===")
    result = m.check_kv_cache_consistency(seq_len=40, key=jax.random.PRNGKey(7), n_new_tokens=6)
    print(f"match={result['match']} match_rate={result['match_rate']}")
    assert result["match"], "generate_kv_cache diverges from generate_no_cache -- generation path broken"

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
