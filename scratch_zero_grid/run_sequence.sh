#!/bin/bash
set -e
cd /Users/muaz/code/qcute

echo "=== [1/3] ks21_overfit10k_v16pq4 training (with mtp_heads=4, eval_decode_mtp_verify) ==="
uv run python -m qcute.qcute_zero.qcute_zero --config configs/qcute_zero/ks21_overfit10k_v16pq4.py --eval_decode_mtp_verify True

echo "=== [2/3] ks1_overfit10k_wavefront2 training (mtp_heads=5, eval_decode_mtp_verify) ==="
uv run python -m qcute.qcute_zero.qcute_zero --config configs/qcute_zero/ks1_overfit10k_wavefront2.py --eval_decode_mtp_verify True

echo "=== [3/3] qualitative wavefront check (ntp / mtp / wavefront-ntp) ==="
uv run python scripts/qual_wavefront_check.py --checkpoint logs/qcute_zero_ks1_overfit10k_wavefront2/last.pt --n_bytes 10000

echo "=== ALL DONE ==="
