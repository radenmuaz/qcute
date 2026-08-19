# qcute v5 status

Reset to a fresh log at this point — the prior narrative predates the modular rewrite below and is
now more archival than actionable. Full history: [docs/archive2/status.md](archive2/status.md)
(pre-modular v5, `qcute_v5_concat.py`/`qcute_v5_stack.py` lineage) and
[docs/archive3/status.md](archive3/status.md) (the immediately-prior log, spanning the qfb fix
through the pre-modular quant-scheme/entropy-reg/per-mode-decode work). New entries go below,
same convention: newest at the bottom, session-dated where useful.

## Where things stand

The standalone `qcute/qcute_v5_concat.py` and `qcute/qcute_v5_stack.py` (dense O(L^2) references)
and the efficient-attention `qcute/qcute_v5.py`/`qcute_v5_concat.py` pair described in the archived
logs have been superseded as the *default* by a modular rewrite:

- **`qcute/qcute_v5_common.py`**: shared `Config`, `LM` (renamed from `Backbone`) class, the three
  `QuantScheme` implementations, attention primitives, training loop, checkpointing, argparser.
- **`qcute/qcute_v5_encoder.py`**: `Encoder` class (PASS 1 — produce a level's code).
- **`qcute/qcute_v5_decoder.py`**: `Decoder` base class plus `ConcatDecoder`/`StackDecoder`
  subclasses (PASS 2 — fused/conditioned prediction, generation, KV-cache, all diagnostics).
- **`qcute/qcute_v5.py`**: `QCuteLM` (renamed from `RefineLM`), the entrypoint — `--decoder_type
  concat|stack` selects which `Decoder` subclass is built.

The old standalone `qcute_v5_concat.py`/`qcute_v5_stack.py` files (pre-modular) are archived under
`qcute/archive3/` — kept for historical reference, not part of active work.

### Quant schemes

Renamed for clearer, technique-neutral geometric naming (the old names conflated the categorical
case with "softmax" and the sign-code case with "BSQ" specifically, when BSQ's own default
projects onto a hypersphere, not the corner-selecting hypercube):

| new | old | geometry |
|---|---|---|
| `SimplexQuant` (`quant_type="simplex"`) | `SoftmaxQuant` (`"softmax"`) | categorical one-hot / softmax simplex |
| `BinaryQuant` (`quant_type="binary"`) | `BSQQuant` (`"bsq"`) | sign code, hypersphere (or hypercube if `binary_lfq=True`) |
| `GridQuant` (`quant_type="grid"`) | `FSQQuant` (`"fsq"`) | axis-aligned integer grid (finite scalar quantization) |

Config fields renamed to match: `binary_bits`/`binary_lfq` (was `bsq_bits`/`bsq_lfq`), `grid_dq`/
`grid_levels`/`grid_bound` (was `fsq_dq`/`fsq_levels`/`fsq_bound`). The underlying math functions
(`gumbel_quantize`, `bsq_quantize`, `fsq_quantize`, `bsq_entropy_reg`, `CodeEmbed`, `FSQEmbed`) keep
their old names deliberately — they name the technique, not the branding that changed.

### `code_sample_mode` replaced by two independent flags (2026-08-18)

The old single `code_sample_mode: "ste"|"sample"|"soft"` enum conflated two orthogonal choices:
whether the code is hardened (discrete + straight-through) and whether stochastic noise is injected
before quantizing. Split into `Config.code_hard: bool = True` / `Config.code_sample: bool = False`
(CLI: `--code_hard true|false`, `--code_sample` store_true):

| old mode | code_hard | code_sample |
|---|---|---|
| `"ste"` (default) | `True` | `False` |
| `"sample"` | `True` | `True` |
| `"soft"` | `False` | `True` |
| *(unreachable before)* | `False` | `False` |

The fourth combination — plain relaxed code, no noise at all — was structurally impossible under
the old enum (`"soft"` always went through the noisy branch) and is the whole point of the split:
it isolates "does hardening matter" from "does noise matter" as independent ablation axes.
`gumbel_quantize`/`bsq_quantize`/`fsq_quantize` all take `(hard, sample)` positionally now instead
of a mode string. Verified via smoke test across all four combinations (`--decoder_type stack`,
Ks=(1,)): `(True,False)`, `(True,True)` via `--code_sample`, `(False,False)`, and `(False,True)`
all train/generate; `check_gen_consistency` is 0/15 for every combo except `(False,True)`
(15/15 mismatched) — expected, since injecting the same Gumbel noise at generation time that was
present during training's teacher-forced forward pass is inherently non-reproducible under greedy
argmax generation, not a bug. `GridQuant.sample_next`'s multinomial-vs-argmax branch and
`BinaryQuant`/`SimplexQuant`'s `sample_next`/`quantize` calls all updated to the new signature.

### Per-level `d_model`/`n_layers`, required quant fields, word presets (modular files only)

Scoped to the new `qcute_v5*` modular files (not the old standalone `qcute_v5_concat.py`/
`qcute_v5_stack.py`, which keep a single global `d_model`/`n_layers`):

- `Config.d_model`/`n_layers` accept either a broadcast scalar (old behavior, unchanged) or an
  explicit per-level tuple, resolved via `resolve_per_level(value, n_levels)` — same pattern
  `attn_window` already used. **No cross-level projection layers were needed**: the code
  representation passed between hierarchy levels lives in *quant-width* space (`V`=`cfg.vocab` for
  simplex, `binary_bits`-wide sign vector, `grid_dq`-wide normalized vector) which is decoupled
  from `d_model` — each level's own `code_embed`/`code_head` bridges quant-width to that level's own
  `D`, so per-level dimension "just worked" once threaded through construction.
- `Config.quant_type`, `vocab`, and the active scheme's code-count param (`binary_bits`/`grid_dq`+
  `grid_levels`) have no default — `build_argparser` raises a clear `p.error(...)` if unset via
  CLI or config file, rather than silently falling back.
- `Config.input_preset`/`output_preset: int` (required, one of `WORD_PRESET_BITS = (1, 4, 8)` — bit/
  nibble/byte) generalize level0's raw alphabet size to `2**preset`, independent of `cfg.vocab`
  (the level1+ code width). Asymmetric input/output presets are conceptually allowed but currently
  raise `NotImplementedError` if unequal — not yet implemented. `unpack_words`/`pack_words` in
  `qcute_v5_common.py` unpack/repack a byte stream into/from `bits`-wide word symbols (`bits==8` is
  an identity no-op, preserving old byte-level behavior exactly).
- Real bug caught while wiring presets: `LM.__init__` was passing the per-level embed-table
  `vocab` (e.g. 16 for level0 under nibble preset) into `quant.init_modules` instead of
  `cfg.vocab`, incorrectly shrinking level0's own code-*production* width. Fixed by using
  `cfg.vocab` unconditionally for `init_modules`, decoupling "input embedding width" (per-level)
  from "code production width" (uniform). Surfaced a second latent bug: `SimplexQuant.embed_for_
  decode`/`embed_input` had been reusing `stage_lm.embed.weight` as the code-embedding table, which
  only worked when every level shared one global vocab — fixed by giving `SimplexQuant` its own
  dedicated `code_embed = nn.Linear(V, D, bias=False)`, matching how `BinaryQuant`/`GridQuant`
  already had dedicated `CodeEmbed`/`FSQEmbed` modules for the same reason.

### Other modular-file features

- `--full_val_eval` (whole-val-set periodic eval, `eval_model_full` with `batch_size=-1`) and
  `--eval_split train|val` for `--eval_only`, mirrored from `qcute.bytelm`.
- Checkpoints now save inside `logs/<run_name>/` instead of a separate `checkpoints/<run_name>/`
  tree (`Checkpointer(args.logs_dir / run_name, ...)`) — `qcute.bytelm` was NOT updated to match,
  still uses the separate `checkpoints/` tree.
- Qualitative-generation pretty-print alignment fix: the `qual_{train,val}_level0_modefull:` log
  line was one space narrower than its sibling `level0_mode{N}:`/`prompt:`/`ground_truth:` lines;
  fixed in `qcute_v5_decoder.py`'s shared logging path (`pad = " " * 5` for the `full` tag vs. `" "
  * 6` for numeric tags), applies to both train and val labels.

### `qcute_v5_stack_noreg` runs (2026-08-18)

No-regularization (`entropy_reg_weight=0.0`) baseline grid on the new modular `qcute_v5.py`
(`--decoder_type stack`), `quant_type="simplex"`, `d_model=256`, `n_layers=1`, `context_len=256`,
8000 steps, full val eval. Logs under `logs/qcute_v5_stack_noreg/<name>/`:

| config | Ks | code_hard/code_sample | best val_bpb |
|---|---|---|---|
| `ks1.py` | `(1,)` | `True`/`False` (ste-equivalent) | 2.7246 |
| `ks21.py` | `(2,1)` | `True`/`False` | 2.8105 |
| `ks221.py` | `(2,2,1)` | `True`/`False` | 2.8414 |
| `ks1_soft.py` | `(1,)` | `False`/`False` (new noise-free-soft combo) | running, 2.4597 at step ~5200/8000 and still improving |

(A stale, incomplete `ks21` attempt from an earlier interrupted session — killed mid-run around
step 4000, never reached `best_val_bpb`-worthy convergence — is kept at
`logs/qcute_v5_stack_noreg/ks21_incomplete_prev/` for reference, not comparable to the table above.)

`bytelm` byte-level baselines on the same enwik8_1M slice, `ctx256`, full val eval, 4000 steps:

| config | best val_bpb |
|---|---|
| `bytelm_xs1_ctx256_fullval` | 2.5846 |
| `bytelm_xs2_ctx256_fullval` | 2.4502 |
| `bytelm_xs4_ctx256_fullval` | 2.4235 |

`bytelm` beats all three `code_hard=True` v5-stack configs, and `ks1` (Ks=(1,), no hierarchy) beats
`ks21`/`ks221` — consistent with the "moving-target/cascade effect" for multi-level configs noted
in the archived logs, now reproduced under the modular rewrite too. Not yet root-caused.

**`ks1_soft` (code_hard=False, code_sample=False) is tracking well below every `code_hard=True`
config, including `bytelm`'s own baselines**, at the same step count it's not yet finished. Leading
hypothesis, **not yet verified — sanity-check before reading into it**: the gain is from removing
hard discretization (STE) entirely rather than from anything about "simplex" specifically —
`code_hard=False` means the code passed downstream is the full continuous softmax distribution, so
there's no straight-through gradient bias and no information loss at the quantization bottleneck
(effectively an ungated soft-attention-style handoff between levels, not a genuine discrete code).
That's a real capability/comparability confound: a `code_hard=False` run isn't testing the same
discrete-code hypothesis the rest of the grid is (no code-usage/entropy stats are meaningful the
same way `to_ids()` bins under `code_hard=True`), so a strong `ks1_soft` val_bpb number is not
evidence that the *discrete* Ks=(1,) config is competitive — it's evidence that discretization
itself carries a cost, which is expected and not new. Needs confirming: (a) let it finish and
compare fairly at matched steps, (b) check whether `ks21_soft`/`ks221_soft` also relax to `bytelm`-
beating numbers (if the hierarchy penalty disappears too under `code_hard=False`, that further
localizes the cascade problem specifically to hard-code straight-through gradients, not to the
hierarchy/cross-attention structure itself).

`qcute_v5_stack_noreg/ks1_overfit1k.py` (n_bytes=1000, val_frac=0.5, 1000 steps) is a fast-
iteration smoke config, not a real training run — best val_bpb 5.3602, not comparable to the table
above.
