"""Label-aware agreement of chunklabel output with PubMed RCT sentence roles.

Layer 2 of research/ideas/evaluation-suite.md. Reuses the raw LLM outputs produced by
benchmarks/fidelity/run.py (no new LLM calls).

    uv run python benchmarks/label_agreement/run.py \
        --data <pubmed-rct>/PubMed_20k_RCT/test.txt --raw <fidelity raw.jsonl>

Reference: contiguous runs of sentences with the same role (background, objective,
methods, results, conclusions) form one chunk labeled with that role.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "fidelity"))
from run import Abstract, load_pubmed, read_records  # noqa: E402

from chunklabel.alignment import align  # noqa: E402
from chunklabel.eval import Agreement, chunk_agreement  # noqa: E402
from chunklabel.postprocess import postprocess  # noqa: E402
from chunklabel.types import Chunk, RawChunk  # noqa: E402

HERE = Path(__file__).parent


def sentence_spans(doc: Abstract) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for s in doc.sentences:
        spans.append((pos, pos + len(s)))
        pos += len(s) + 1
    return spans


def chunks_from_sentence_groups(doc: Abstract, groups: list[tuple[int, int, str]]) -> list[Chunk]:
    """Chunks covering sentence index ranges [i, j) with the given category."""
    spans, text = sentence_spans(doc), doc.text
    return [
        Chunk(category=cat, quote=text[spans[i][0]:spans[j - 1][1]],
              start=spans[i][0], end=spans[j - 1][1])
        for i, j, cat in groups
    ]


def reference_chunks(doc: Abstract) -> list[Chunk]:
    groups: list[tuple[int, int, str]] = []
    for k, label in enumerate(doc.labels):
        if groups and groups[-1][2] == label and groups[-1][1] == k:
            groups[-1] = (groups[-1][0], k + 1, label)
        else:
            groups.append((k, k + 1, label))
    return chunks_from_sentence_groups(doc, groups)


def baselines(doc: Abstract) -> dict[str, list[Chunk]]:
    n = len(doc.sentences)
    return {
        "baseline: whole abstract": chunks_from_sentence_groups(doc, [(0, n, "all")]),
        "baseline: every sentence": chunks_from_sentence_groups(
            doc, [(k, k + 1, f"s{k}") for k in range(n)]),
        "baseline: gold role per sentence": chunks_from_sentence_groups(
            doc, [(k, k + 1, lab) for k, lab in enumerate(doc.labels)]),
    }


def predicted_chunks(record: dict[str, Any], doc: Abstract) -> list[Chunk]:
    raws = [RawChunk(category=c, quote=q) for c, q in record["chunks"]]
    spans = align(raws, doc.text, threshold=85, on_error="skip")
    return postprocess(raws, spans, doc.text)


def mean_agreement(items: list[Agreement]) -> dict[str, float]:
    return {k: float(np.mean([getattr(a, k) for a in items]))
            for k in ("homogeneity", "completeness", "v_measure", "ari")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", type=Path, default=HERE / "results.json")
    args = ap.parse_args()

    docs = {d.pmid: d for d in load_pubmed(args.data, args.n, args.seed)}
    refs = {k: reference_chunks(d) for k, d in docs.items()}
    rows: list[dict[str, Any]] = []

    def evaluate(name: str, by: str, get: Callable[[str], list[Chunk] | None]) -> None:
        scores, n_chunks, n_cats = [], [], []
        for pmid, doc in docs.items():
            pred = get(pmid)
            if pred is None:
                continue
            content = [c for c in pred if c.quote.strip()]
            n_chunks.append(len(content))
            n_cats.append(len({c.category for c in content}))
            scores.append(chunk_agreement(pred, refs[pmid], doc.text, by=by))  # type: ignore[arg-type]
        row = {"method": name, "by": by, "docs": len(scores),
               "chunks_per_doc": float(np.mean(n_chunks)),
               "categories_per_doc": float(np.mean(n_cats)), **mean_agreement(scores)}
        rows.append(row)
        print(f"{name:34s} by={by:8s} docs={len(scores):2d} chunks={row['chunks_per_doc']:5.1f} "
              f"cats={row['categories_per_doc']:5.1f}  hom={row['homogeneity']:.3f} "
              f"com={row['completeness']:.3f} V={row['v_measure']:.3f} ARI={row['ari']:.3f}")

    print(f"{len(docs)} abstracts; reference segments/doc "
          f"{np.mean([len(r) for r in refs.values()]):.1f}, "
          f"distinct roles/doc {np.mean([len(set(d.labels)) for d in docs.values()]):.1f}")
    for name in baselines(next(iter(docs.values()))):
        for by in ("category", "chunk"):
            evaluate(name, by, lambda pmid, name=name: baselines(docs[pmid])[name])

    records = [r for r in read_records(args.raw) if r["doc"] in docs and "error" not in r]
    for model, mode in sorted({(r["model"], r["mode"]) for r in records}):
        by_doc = {r["doc"]: r for r in records if (r["model"], r["mode"]) == (model, mode)}
        # two_pass raw output has boundaries only (labels come from a later LLM step).
        for by in (("category", "chunk") if mode == "one_pass" else ("chunk",)):
            evaluate(f"{model} {mode}", by,
                     lambda pmid, m=by_doc: predicted_chunks(m[pmid], docs[pmid])
                     if pmid in m else None)

    args.results.write_text(json.dumps(
        {"n_abstracts": len(docs), "seed": args.seed, "unit": "non-whitespace characters",
         "rows": rows}, indent=2))
    print(f"wrote {args.results}")


if __name__ == "__main__":
    main()
