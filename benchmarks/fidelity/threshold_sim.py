"""Simulate alignment scores to choose a default fuzzy threshold (no LLM needed).

Positives: real sentences of an abstract with controlled edits, aligned against that
abstract. They should be accepted.
Negatives: sentences from *other* abstracts (same domain), aligned against the abstract.
They should be rejected.

Compares chunklabel's original scorer ("current" below: fuzz.ratio against a window of
len(quote) + 20) with fuzz.partial_ratio_alignment, which chunklabel uses since this
study, by quote length.

    uv run python benchmarks/fidelity/threshold_sim.py \
        --data <pubmed-rct>/PubMed_20k_RCT/test.txt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run import load_pubmed  # noqa: E402


HERE = Path(__file__).parent
LENGTH_BINS = [(0, 40), (40, 80), (80, 160), (160, 10_000)]


def char_edits(s: str, k: int, rng: random.Random) -> str:
    chars = list(s)
    for _ in range(k):
        i = rng.randrange(len(chars))
        if rng.random() < 0.5:
            del chars[i]
        else:
            chars[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
        if not chars:
            break
    return "".join(chars)


def drop_word(s: str, rng: random.Random) -> str:
    words = s.split()
    if len(words) > 2:
        del words[rng.randrange(1, len(words) - 1)]
    return " ".join(words)


def insert_words(s: str, rng: random.Random) -> str:
    words = s.split()
    words.insert(rng.randrange(1, max(2, len(words))), "significantly and notably")
    return " ".join(words)


def current_score(q: str, text: str) -> float:
    """The scorer chunklabel used before this study (fuzz.ratio against a window of
    len(q) + 20 characters), kept here so the comparison stays reproducible."""
    from rapidfuzz import fuzz

    if q in text:
        return 100.0
    window = len(q) + 20
    return max(
        fuzz.ratio(q, text[i : i + window]) for i in range(max(1, len(text) - len(q) + 1))
    )


def partial_score(q: str, text: str) -> float:
    from rapidfuzz import fuzz

    if q in text:
        return 100.0
    r = fuzz.partial_ratio_alignment(q, text)
    return r.score if r is not None else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200, help="abstracts to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    docs = load_pubmed(args.data, args.n, args.seed + 1)

    perturb = {
        "1 char edit": lambda s: char_edits(s, 1, rng),
        "3 char edits": lambda s: char_edits(s, 3, rng),
        "5% char edits": lambda s: char_edits(s, max(1, len(s) // 20), rng),
        "drop 1 word": lambda s: drop_word(s, rng),
        "insert 3 words": lambda s: insert_words(s, rng),
    }
    scores: dict[tuple[str, str, int], list[float]] = {}

    def add(kind: str, scorer: str, q: str, value: float) -> None:
        b = next(i for i, (lo, hi) in enumerate(LENGTH_BINS) if lo <= len(q) < hi)
        scores.setdefault((kind, scorer, b), []).append(value)

    for j, doc in enumerate(docs):
        text = doc.text
        for s in rng.sample(doc.sentences, min(3, len(doc.sentences))):
            for kind, f in perturb.items():
                q = f(s)
                if q in text:
                    continue
                add(kind, "current", q, current_score(q, text))
                add(kind, "partial", q, partial_score(q, text))
        other = docs[(j + 1) % len(docs)]
        for q in rng.sample(other.sentences, min(3, len(other.sentences))):
            add("NEGATIVE other abstract", "current", q, current_score(q, text))
            add("NEGATIVE other abstract", "partial", q, partial_score(q, text))

    kinds = list(perturb) + ["NEGATIVE other abstract"]
    out = []
    for scorer in ("current", "partial"):
        print(f"\n== scorer: {scorer}  (median [5th-95th pct] by quote length)")
        print(f"{'':26s}" + "".join(f"{f'len {lo}-{hi}':>22s}" for lo, hi in LENGTH_BINS))
        for kind in kinds:
            cells = []
            for b in range(len(LENGTH_BINS)):
                v = scores.get((kind, scorer, b), [])
                if v:
                    p5, p50, p95 = np.percentile(v, [5, 50, 95])
                    cells.append(f"{p50:5.1f} [{p5:4.1f}-{p95:5.1f}] n={len(v):3d}")
                    out.append({"scorer": scorer, "kind": kind, "len_bin": LENGTH_BINS[b],
                                "n": len(v), "p5": p5, "p50": p50, "p95": p95})
                else:
                    cells.append("-")
            print(f"{kind:26s}" + "".join(f"{c:>22s}" for c in cells))
        for t in (60, 70, 75, 80, 85, 90):
            pos = [x for (k, sc, _), v in scores.items() if sc == scorer and not
                   k.startswith("NEG") for x in v]
            neg = [x for (k, sc, _), v in scores.items() if sc == scorer and
                   k.startswith("NEG") for x in v]
            print(f"  threshold {t}: positives accepted {np.mean(np.array(pos) >= t):.3f}, "
                  f"negatives accepted {np.mean(np.array(neg) >= t):.3f}")
    (HERE / "threshold_sim.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
