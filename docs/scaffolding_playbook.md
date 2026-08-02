# Scaffolding playbook

A reusable, detailed prompt for bootstrapping a new uv-managed, design-doc-
driven ML research project the way *this* repo (qcute) is actually built —
not just the initial skeleton, but the experiment-running conventions it
converged on. Hand this to a fresh Claude Code session (swap in the new
design doc, package name, and dataset) to reproduce the structure and style.

---

Set up this repo as a uv-managed Python project implementing the design in
`<DESIGN_DOC>.md`. Match this exact structure and these conventions:

## 1. Environment

`uv init --package --python 3.12`. If a package dir already exists,
consolidate into it (flat layout, not `src/`) — set
`[tool.uv.build-backend] module-root = ""` for flat layout, or
`[tool.uv] package = false` if this is just runnable scripts, not something
meant to be installed/importable elsewhere (qcute uses `package = false`).

## 2. `.gitignore` up front

venv, caches, build artifacts, and every directory that will fill up with
run artifacts: `datasets/` (downloaded/derived data), `logs/` (training
logs — see §6). Add these *before* the first training run, not after.

## 3. One module per component, named after its main class

Implement each major component from the design doc as its own module
inside the package — e.g. `qcute/bytelm.py` defines `class ByteLM`,
`qcute/qcutelm.py` defines `class QCuteLM`. **The module filename should
match the primary class it defines** (lowercased) — this makes `from
pkg.bytelm import ByteLM` self-explanatory and avoids the alternative of
generic names like `model.py` that don't survive a second model being
added later.

Never leave a bare top-level script with the same name as the package dir
(e.g. `foo.py` next to `foo/` silently breaks `import foo` for anything
else in the repo). Each module should be genuinely self-contained — no
imports between sibling modules unless there's a concrete reuse need (qcute's
`bytelm.py` and `qcutelm.py` duplicate a small `Logger` class and
`load_enwik8`/`lr_at` helpers rather than share them, on purpose: it keeps
each file readable standalone and avoids premature coupling between things
that are still actively changing). Run everything via `python -m pkg.module`.

Superseded implementations get moved to `archive/<descriptive_name>.py`
(not deleted) with a one-line header explaining what replaced them and why
— `git mv`, not `rm`, so history is preserved.

## 4. Every training script's `main()` follows the same shape

This is the part worth copying exactly, because it's what makes running
many small experiments tractable:

**a. Two-tier config: Python config file + CLI override.**
Add a `--config path/to/file.py` flag, parsed in a pre-pass (`parse_known_args`
on a small parser with just `--config`) before building the full argument
parser. Load the config file as a plain Python module (`importlib.util`,
not YAML/JSON — real Python means `Path(...)`, arithmetic, and comments are
free) and feed its module-level variables into `parser.set_defaults(...)`,
filtered to known argument names. Then do the real `parser.parse_args()`.
Net effect: **config file values override the script's hardcoded defaults,
but explicit CLI flags override the config file.** This gives you
reproducible named experiments (`configs/bytelm_xs_tiny_longrun.py`) without
losing the ability to override one flag ad hoc for a quick variant.
Config files live in `configs/`, one per named experiment, with a docstring
explaining what the experiment tests and the exact invocation.

**b. Log-file naming falls back through three tiers**, most to least specific:
explicit `--log_file` path → else `logs/<config_file_stem>_<timestamp>.log`
if `--config` was given → else a formula from the CLI args themselves (e.g.
`logs/qcute_lm_<preset>_<timestamp>.log`). Implement this as a plain
if/elif/else in `main()`, not magic.

**c. Every run gets a `tqdm` progress bar *and* two logfiles**, both written
only at `--log_every`/`--eval_every` intervals — never on every step, and
never touched by tqdm's live `\r`-redraws:
  - `<name>.log` — raw human-readable text, exactly what's printed to the
    terminal (via `tqdm.write()`, which doesn't clobber the progress bar).
    `tail -f` this from another terminal to watch a background run live.
  - `<name>.jsonl` — one JSON object per line, same information
    structured, for later plotting/analysis.

Every logged line is prefixed with elapsed time as `[HH:MM:SS]` (tracked
from when the `Logger` was constructed), and the JSON record carries
`elapsed_s` (int) and `elapsed_hms` alongside the metrics — not a raw Unix
epoch timestamp, which isn't human-readable and isn't what anyone actually
wants when reading a log back later.

Implement this as a small `Logger` class, duplicated per-script like
everything else in §3:

```python
class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.text_path = path.with_suffix(".log")
        self.json_path = path.with_suffix(".jsonl")
        self.text_f = open(self.text_path, "a")
        self.json_f = open(self.json_path, "a")
        self.start_time = time.time()

    def __call__(self, msg: str, **record) -> None:
        elapsed_s = int(time.time() - self.start_time)
        line = f"[{format_hms(elapsed_s)}] {msg}"
        tqdm.write(line)
        self.text_f.write(line + "\n"); self.text_f.flush()
        self.json_f.write(json.dumps({"elapsed_s": elapsed_s, "msg": msg, **record}) + "\n")
        self.json_f.flush()
```

**d. Train/val split + periodic eval, always**, even for a quick script —
`split_train_val(data, val_frac)` and an `eval_*()` function run every
`--eval_every` steps on a held-out slice. Report both train-step metrics
(noisy, every `--log_every`) and val metrics (averaged over `--eval_batches`,
less noisy) — the gap between them is often the most important signal (see
§7).

## 5. Data prep is its own idempotent script

`scripts/prepare_data.py`: downloads the full dataset if missing (skips if
present, `--force` to re-fetch), and cuts a small `_tiny` variant (e.g.
500,000 bytes) for fast local iteration — every training script accepts
`--data path/to/either` so the same code path exercises both scales.
Running the script twice does nothing destructive the second time.

## 6. Long/background runs: `run_in_background` + a `Monitor` filter, never a sleep-poll loop

For anything that'll run more than ~2 minutes: launch it, let it move to
background automatically, then attach a `Monitor` that tails the *text*
logfile and greps for the lines worth surfacing (e.g. `val_bpb` checkpoints)
**plus failure signatures** (`Traceback|Error|Killed`) so a crash isn't
silently missed. Never `sleep`-loop and poll manually — either wait for the
background task's own completion notification, or let the Monitor's
grep-filtered tail deliver periodic updates. If you need to change
hyperparameters mid-run, there's no checkpoint/resume in scripts this
simple — stop the run and relaunch with a bigger step budget; because the
LR schedule is stateless (see §7), this is equivalent to having let the
original run continue, not a wasted restart.

## 7. LR schedule: linear warmup, then constant — deliberately, for comparability

Use `lr_at(step, warmup, peak) = peak * step/warmup` during warmup, `peak`
after — not cosine decay. This is a real methodological choice, not just
"simpler": a non-decaying LR is the standard way to *expose* whether a
model is overfitting a small dataset (train loss keeps dropping while val
rises, sometimes non-monotonically — this is the empirical signature of
epoch-wise double descent, Nakkiran et al. 2019, and it's specifically
easiest to see under a constant LR because decay would suppress it). Use
the exact same schedule function/params across every script being compared
head-to-head — otherwise a BPB difference could just be an LR-schedule
artifact, not a real architectural finding.

## 8. `docs/` structure

- The design doc itself, moved in verbatim.
- `architecture.md`: maps code → doc sections, one `##` per module,
  documents load-bearing non-obvious fixes (`"don't remove this init
  scaling, default init blows up the loss to ~1000 bits/byte"`), and a
  "known gaps vs. the full design" section at the end.
- `status.md`: phase-by-phase checklist against the design doc's own phase
  plan, updated as work lands — not aspirational, an honest record of
  what's actually been run vs. just implemented-but-untested.
- This file.

## 9. `CLAUDE.md`: lean, always

Commands block + a two-sentence architecture pointer into
`docs/architecture.md`. Never duplicate `architecture.md`'s content here —
one is the fast-glance summary, the other is the detail; if they drift the
lean one wins on staleness risk.

## 10. Workflow policy

Implement continuously without pausing for review checkpoints — the user
watches diffs/terminal output live and interrupts to redirect. Smoke-test
every script with tiny `--steps`/`--n_bytes` before calling it done — this
is how a bad default init producing nonsense loss on step 1 gets caught
immediately instead of after a 30-minute run. Never `git commit`/`git push`
unless explicitly asked for that specific commit/push.
