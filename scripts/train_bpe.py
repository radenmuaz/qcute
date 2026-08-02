"""Train a byte-level BPE tokenizer (sentencepiece) on enwik8/enwik8_tiny.

    uv run python scripts/train_bpe.py --data datasets/enwik8_tiny.gz --vocab_size 8192

Produces datasets/bpe_<stem>_<vocab_size>.{model,vocab}. sentencepiece needs
a plain-text file, so the raw byte corpus is decoded utf-8 (errors=replace
— enwik8 is a Wikipedia XML dump, effectively all valid UTF-8, so this loses
~nothing in practice, but it does mean exact-byte roundtrip isn't
guaranteed for the rare invalid sequence; see qcute/bpelm.py's bpb note).

vocab_size default (8192, kept power-of-2) reaches ~3.9 bytes/token on the
450,000-byte tiny training subset; larger vocabs buy a bit more (~4.8
bytes/token around 32768) before returns invert into corpus-specific
phrase-memorization on a corpus this small — a corpus-size ceiling, not a
config knob to push through.

The target here is ~4 bytes/timestep, not 8: typical byte-level BPE scaling
on English text is logarithmic-ish in vocab size (GPT-2's 50k vocab
averages ~4 bytes/token; even much larger production vocabs, 100k-256k,
typically land ~4-5, not 8). Reaching 8 bytes/token would need either a
vocab in the hundreds of thousands or a corpus dominated by repetitive
multi-word boilerplate — plausibly closer on the full enwik8.gz corpus
given its wiki-markup boilerplate (`[[Category:`, `http://`, XML tags), but
"closer to 5-6" is the honest expectation, not "closer to 8". qcute.bytelm's
`xs` preset and qcute.qcutelm's `K` are both calibrated to 4 for this
reason (see qcute/bytelm.py's PRESETS comment).
"""
from __future__ import annotations

import argparse
import gzip
import tempfile
from pathlib import Path

import sentencepiece as spm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_tiny.gz"))
    p.add_argument("--n_bytes", type=int, default=None, help="prefix of the corpus to train on (default: all)")
    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--out_dir", type=Path, default=Path("datasets"))
    args = p.parse_args()

    with gzip.open(args.data, "rb") as f:
        raw = f.read(args.n_bytes) if args.n_bytes else f.read()
    text = raw.decode("utf-8", errors="replace")

    model_prefix = args.out_dir / f"bpe_{args.data.stem}_{args.vocab_size}"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
        corpus_path = f.name

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=str(model_prefix),
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=-1, bos_id=-1, eos_id=-1, unk_id=0,
        # Lossless byte-level tokenization — sentencepiece's NLP-oriented defaults
        # (NFKC normalization, whitespace collapsing) silently drop information
        # (e.g. distinct \n vs " " become indistinguishable), which breaks any
        # downstream exact bpb claim. identity normalization + no whitespace
        # collapsing + byte_fallback (so any uncovered char still round-trips
        # via raw byte tokens instead of lossy <unk>) makes encode/decode a
        # true bijection with the original byte stream.
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        byte_fallback=True,
        add_dummy_prefix=False,  # no synthetic leading space — every ▁ in every
                                  # piece then corresponds to a real byte, with
                                  # no first-token special case needed downstream
    )
    print(f"wrote {model_prefix}.model / {model_prefix}.vocab")

    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    sample = text[:200_000]
    ids = sp.encode(sample)
    n_bytes_sample = len(sample.encode("utf-8"))
    roundtrip_ok = sp.decode(ids) == sample
    print(f"sample: {n_bytes_sample} bytes -> {len(ids)} tokens ({n_bytes_sample/len(ids):.2f} bytes/token)")
    print(f"lossless roundtrip check: {'PASS' if roundtrip_ok else 'FAIL'}")
    if not roundtrip_ok:
        raise SystemExit("tokenizer is not lossless — bpb accounting in qcute.bpelm would be invalid; investigate before using this .model")


if __name__ == "__main__":
    main()
