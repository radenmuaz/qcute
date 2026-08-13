"""qcute.qcute_refine — always aliases the LATEST qcute_refine_vN.py.

Convention (session ask): this file has no version suffix precisely so it
always points at current work — callers, scripts, and docs that want
"whatever the best qcute_refine architecture is right now" import/run
`qcute.qcute_refine` instead of tracking which vN is current by hand.
Historical/comparison work should still target a specific `qcute_refine_
vN.py` directly (as scripts/probe_decoder_kv_contribution.py,
scripts/bench_bit_heads.py, and every config's own docstring already do) —
this file is for "give me the latest," not for reproducibility pinning.

Currently: qcute_refine_v4.py (EncoderLevel fusion, no DecoderLevel — see
its own module docstring for the full architecture and rationale).

To promote a new version: change the import line below to the new vN.
Nothing else in this file should ever need to change.

    uv run python -m qcute.qcute_refine --config configs/qcute_refine_v4_pq.py
"""
from qcute.qcute_refine_v4 import *  # noqa: F401,F403
from qcute.qcute_refine_v4 import main

if __name__ == "__main__":
    main()
