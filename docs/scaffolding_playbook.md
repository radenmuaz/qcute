# Scaffolding playbook

A reusable prompt for bootstrapping a new uv-managed, design-doc-driven Python
project the way this repo was set up. Hand this to a fresh Claude Code session
(swap in the new design doc and package name) to reproduce the pattern.

---

Set up this repo as a uv-managed Python project implementing the design in
`<DESIGN_DOC>.md`.

1. **Env**: `uv init --package --python 3.12`. If a package dir already
   exists, consolidate into it (flat layout, not `src/`) — set
   `[tool.uv.build-backend] module-root = ""` for flat layout, or
   `[tool.uv] package = false` if this is just runnable scripts, not
   something meant to be installed/importable elsewhere.
2. **`.gitignore`**: venv, caches, build artifacts, and any large data dir
   (e.g. `datasets/`) up front.
3. **Module naming**: implement each major component from the design doc as
   its own module inside the package. Never leave a bare top-level script
   with the same name as the package dir (e.g. `foo.py` next to `foo/`
   silently breaks `import foo` for anything else in the repo). Pick
   distinct module names, run via `python -m pkg.module`.
4. **`docs/`**: move the design doc there, add `architecture.md` (maps
   code → doc sections, flags load-bearing non-obvious fixes — e.g. "don't
   remove this init scaling, default init blows up the loss") and
   `status.md` (phase-by-phase go/no-go tracking against the doc's plan).
5. **`CLAUDE.md`**: lean — commands plus a pointer to `docs/architecture.md`,
   never a duplicate of it.
6. **`README.md`**: Quickstart (`uv sync`, dataset `curl`, run commands) + a
   short "So far" status summary, linking into `docs/` for detail.
7. **Smoke-test everything** with tiny steps/data before calling it done —
   this is how you catch things like a bad default init producing nonsense
   loss on step 1.
8. **Workflow policy**: implement continuously without pausing for review
   checkpoints; never `git commit`/`git push` unless explicitly asked for
   that specific commit/push.
