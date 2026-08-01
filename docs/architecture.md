# Architecture

Two self-contained modules in the `qcute/` package: `qcute/lm.py` and
`qcute/tokenizer.py`. Neither imports the other or shares a common name with
the package itself (an earlier top-level `qcute.py` next to the `qcute/`
package collided on `import qcute` — renamed; `qcute/lm.py` inlines its own
tiny `load_enwik8` rather than importing it from `qcute/tokenizer.py`, to
keep the two independent). Merge them once a third module needs to reuse
encoder/decoder/LM pieces.

See [continuous_tokenizer_handover.md](continuous_tokenizer_handover.md) for
the full design this implements, and [status.md](status.md) for what's done.

## `qcute/lm.py` — byte-level baseline LM

The Phase 0 "number to beat" (handover §5, Phase 0): a plain causal
transformer trained directly on raw bytes, no tokenizer.

- `ByteLM`: pre-norm transformer blocks, RoPE on Q/K (`rope_cos_sin` /
  `apply_rope`), `F.scaled_dot_product_attention(is_causal=True)`,
  GELU MLP (4x hidden), weight-tied embedding/output head.
- Init: GPT-2-style — `N(0, 0.02)` for all `Linear`/`Embedding` weights, with
  the residual-stream projections (`attn.out`, `mlp.down`) additionally scaled
  by `1/sqrt(2 * n_layers)`. **Load-bearing**: PyTorch's default `Embedding`
  init (`std=1`) blows up logits at init to ~1000 bits/byte instead of the
  correct ~8 (`log2(256)`); don't remove this without re-verifying init-time
  bpb is sane.
- `bits_per_byte()`: exact BPB from softmax cross-entropy over the 256-way
  byte vocab — no ELBO needed, unlike the tokenizer's quantized bottleneck.
- Presets in `PRESETS` (power-of-2-friendly dims): `sd` ≈101M
  (d=1024, 8 layers, 16 heads, ctx 2048), `md` ≈403M (d=2048, 8 layers,
  16 heads, ctx 2048). Param count ≈ `12 * d_model^2 * n_layers`
  (vocab is only 256, so embedding params are negligible).

## `qcute/tokenizer.py` — Phase 1 tokenizer autoencoder

Standalone encoder+decoder (no LM yet) — validates the bottleneck per the
Phase 1 go/no-go: reconstruction accuracy > 99.5% on held-out bytes at K=8.

- `FSQ` (handover §1.2.2): bounded + rounded straight-through bottleneck,
  `dq=6` dims × `L=8` levels ⇒ implicit codebook `8^6`.
- `CausalByteEncoder` (handover §1.3): causal-over-bytes body emitting one
  latent every `K` bytes. Uses `nn.GRU` as a stand-in for the doc's
  recommended Mamba-style SSM — both are causal, O(N) recurrent bodies;
  swap in a real SSM kernel if GRU throughput/quality becomes limiting.
- `NATDecoder` (handover §1.4.2/1.4.3a): memoryless — conditions only on
  `z_t` via per-layer FiLM into a small bidirectional transformer, one-shot
  factorized training (independent per-position cross-entropy, no MaskGIT
  masking yet). This is explicitly the Phase 1 baseline decoder; the
  streaming-SSM decoder (handover §1.4.1) is a Phase 2+ upgrade.
- Config defaults match the handover's "compact, text-only" row: `K=8`,
  `dq=6`, `L=8`.

## Known gaps vs. the full design (tracked in [status.md](status.md))

- No LM over latents yet (Phase 2) — `qcute/tokenizer.py` only trains the
  autoencoder.
- Decoder training is one-shot factorized, not the recommended time-free
  masked diffusion (handover §1.4.3c) for a proper BPB ELBO.
- No geometric-state mixers (Phase 3) — `qcute/lm.py`'s attention is plain
  softmax.
