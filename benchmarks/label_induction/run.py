"""ZeroShotPromptSegmenter.fit on PubMed 20k RCT abstracts (llama.cpp). Issue #18.

fit's first step (schema-free split of every text) is the one_pass mode of
benchmarks/fidelity, so its raw outputs are replayed instead of calling the LLM again.
The LLM is called for Normalizer (once per model) and for predict (once per text).

    # 1. fit + predict (expensive; predict is resumable)
    uv run --extra llamacpp python benchmarks/label_induction/run.py generate \
        --data <pubmed-rct>/PubMed_20k_RCT/test.txt --raw <fidelity raw.jsonl> \
        --model <path.gguf> --name qwen2.5-7b --out <outside the repo>/induction

    # 2. report (cheap)
    uv run python benchmarks/label_induction/run.py report \
        --data <pubmed-rct>/PubMed_20k_RCT/test.txt --out <same>/induction

Outputs with abstract text stay outside the repository. Only aggregates and the
induced label sets are committed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from chunklabel import ZeroShotPromptSegmenter
from chunklabel.eval import chunk_agreement, fidelity
from chunklabel.llm.base import LLMBackend
from chunklabel.types import Chunk, RawChunk

HERE = Path(__file__).parent
ROOT = HERE.parent


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fid = _load("fidelity_run", ROOT / "fidelity" / "run.py")
agr = _load("label_agreement_run", ROOT / "label_agreement" / "run.py")


class ReplayBackend(LLMBackend):
    """extract_chunks returns recorded one_pass outputs; other calls go to ``inner``."""

    def __init__(self, recorded: dict[str, list[RawChunk]], inner: LLMBackend | None) -> None:
        self.recorded = recorded
        self.inner = inner
        self.last_raw: list[RawChunk] = []

    def extract_chunks(self, text: str) -> list[RawChunk]:
        return self.recorded[text]

    def extract_labeled_chunks(self, text: str, labels: Any) -> list[RawChunk]:
        assert self.inner is not None
        self.last_raw = self.inner.extract_labeled_chunks(text, labels)
        return self.last_raw


def generate(args: argparse.Namespace) -> None:
    from llama_cpp import Llama

    from chunklabel.llm.backend import ClientBackend
    from chunklabel.llm.client import LlamaCppClient

    docs = fid.load_pubmed(args.data, args.n, args.seed)
    by_pmid = {r["doc"]: r for r in fid.read_records(args.raw)
               if r["model"] == args.name and r["mode"] == "one_pass" and "error" not in r}
    docs = [d for d in docs if d.pmid in by_pmid]
    recorded = {d.text: [RawChunk(c, q) for c, q in by_pmid[d.pmid]["chunks"]] for d in docs}
    args.out.mkdir(parents=True, exist_ok=True)

    client = LlamaCppClient(Llama(model_path=str(args.model), n_ctx=args.n_ctx,
                                  n_gpu_layers=-1, verbose=False))
    backend = ReplayBackend(recorded, ClientBackend(client))

    fit_path = args.out / f"fit_{args.name}.json"
    if fit_path.exists():
        fitted = json.loads(fit_path.read_text())
    else:
        seg = ZeroShotPromptSegmenter(client=client, backend=backend, on_align_error="skip")
        t0 = time.time()
        seg.fit([d.text for d in docs])
        fitted = {"docs": [d.pmid for d in docs], "labels_": seg.labels_,
                  "mapping_": seg.mapping_, "segmentations_": seg.segmentations_,
                  "seconds": round(time.time() - t0, 1)}
        fit_path.write_text(json.dumps(fitted, indent=1))
    print(f"{args.name}: {len(fitted['mapping_'])} free labels -> "
          f"{len(fitted['labels_'])} labels ({fitted['seconds']}s)", flush=True)

    seg = ZeroShotPromptSegmenter(labels=fitted["labels_"], client=client, backend=backend,
                                  on_align_error="skip")
    pred_path = args.out / f"predict_{args.name}.jsonl"
    done = {r["doc"] for r in fid.read_records(pred_path)}
    for j, doc in enumerate(docs):
        if doc.pmid in done:
            continue
        t0 = time.time()
        rec: dict[str, Any] = {"doc": doc.pmid}
        try:
            [rec["segments"]] = seg.predict([doc.text])
            rec["raw"] = [[c.category, c.quote] for c in backend.last_raw]
        except Exception as e:  # noqa: BLE001 - record and continue
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        rec["seconds"] = round(time.time() - t0, 1)
        with pred_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  {args.name} predict {j + 1}/{len(docs)} {rec['seconds']}s "
              f"{rec.get('error', 'ok')[:60]}", flush=True)


def to_chunks(segments: list[list[Any]], text: str) -> list[Chunk]:
    return [Chunk(category=lab, quote=text[s:e], start=s, end=e) for s, e, lab in segments]


def report(args: argparse.Namespace) -> None:
    docs = {d.pmid: d for d in fid.load_pubmed(args.data, args.n, args.seed)}
    rows: list[dict[str, Any]] = []
    label_sets: dict[str, list[str]] = {}
    for fit_path in sorted(args.out.glob("fit_*.json")):
        name = fit_path.stem.removeprefix("fit_")
        fitted = json.loads(fit_path.read_text())
        label_sets[name] = fitted["labels_"]
        preds = {r["doc"]: r for r in fid.read_records(args.out / f"predict_{name}.jsonl")}
        errors = sum("error" in r for r in preds.values())

        def row(step: str, segs: dict[str, list[list[Any]]], raws: dict[str, Any] | None,
                name: str = name) -> None:
            scores, n_seg, n_cat, gap, exact, quotes = [], [], [], [], 0, 0
            for pmid, segments in segs.items():
                doc = docs[pmid]
                pred = to_chunks(segments, doc.text)
                ref = agr.reference_chunks(doc)
                scores.append(chunk_agreement(pred, ref, doc.text, by="category"))
                n_seg.append(len(pred))
                n_cat.append(len({c.category for c in pred}))
                if raws is not None:
                    rep = fidelity([RawChunk(c, q) for c, q in raws[pmid]], doc.text)
                    gap.append(rep.gap_coverage)
                    exact += rep.n_exact
                    quotes += rep.n_quotes
            r: dict[str, Any] = {
                "model": name, "step": step, "docs": len(segs),
                "free_labels": len(fitted["mapping_"]), "labels": len(fitted["labels_"]),
                "segments_per_doc": float(np.mean(n_seg)),
                "labels_per_doc": float(np.mean(n_cat)),
                **agr.mean_agreement(scores)}
            if raws is not None:
                r |= {"quotes": quotes, "verbatim": exact / quotes,
                      "gap_coverage": float(np.mean(gap)), "errors": errors}
            rows.append(r)
            print(f"{name:12s} {step:8s} docs={r['docs']:2d} labels={r['labels']:3d} "
                  f"(free {r['free_labels']}) seg/doc={r['segments_per_doc']:4.1f} "
                  f"labels/doc={r['labels_per_doc']:4.1f} hom={r['homogeneity']:.3f} "
                  f"com={r['completeness']:.3f} ARI={r['ari']:.3f}"
                  + (f" verbatim={r['verbatim']:.3f} gap={r['gap_coverage']:.3f} "
                     f"err={errors}" if raws is not None else ""))

        row("fit", dict(zip(fitted["docs"], fitted["segmentations_"])), None)
        ok = {k: r for k, r in preds.items() if "error" not in r}
        row("predict", {k: r["segments"] for k, r in ok.items()},
            {k: r["raw"] for k, r in ok.items()})

    args.results.write_text(json.dumps(
        {"n_abstracts": len(docs), "seed": args.seed, "rows": rows,
         "label_sets": label_sets}, indent=2))
    print(f"wrote {args.results}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "report"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--n", type=int, default=40)
        p.add_argument("--seed", type=int, default=0)
        if name == "generate":
            p.add_argument("--raw", type=Path, required=True)
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--name", required=True)
            p.add_argument("--n-ctx", type=int, default=8192)
        else:
            p.add_argument("--results", type=Path, default=HERE / "results.json")
    args = ap.parse_args()
    generate(args) if args.cmd == "generate" else report(args)


if __name__ == "__main__":
    main()
