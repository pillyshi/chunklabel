"""FewShotPromptSegmenter on PubMed 20k RCT abstracts (llama.cpp). Issue #19.

The example pool is a sample of the train split, labeled with the gold roles
(contiguous sentences with the same role form one segment). Test abstracts are the same
40 as in benchmarks/fidelity and benchmarks/label_agreement.

    # 1. generate (expensive; resumable)
    uv run --extra llamacpp python benchmarks/few_shot/run.py generate \
        --data <pubmed-rct>/PubMed_20k_RCT/test.txt --train <...>/train.txt \
        --model <path.gguf> --name qwen2.5-7b --configs random:3,similar:3 \
        --out <outside the repo>/few_shot.jsonl

    # 2. report (cheap)
    uv run python benchmarks/few_shot/run.py report \
        --data <pubmed-rct>/PubMed_20k_RCT/test.txt --out <same>/few_shot.jsonl

Outputs contain abstract text, so they stay outside the repository. Only aggregates
are committed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from chunklabel import FewShotPromptSegmenter
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


def gold_segmentation(doc: Any) -> list[tuple[int, int, str]]:
    return [(c.start, c.end, c.category.lower()) for c in agr.reference_chunks(doc)]


class RecordingBackend(LLMBackend):
    """Passes calls through and keeps the last raw output, for fidelity."""

    def __init__(self, inner: LLMBackend) -> None:
        self.inner = inner
        self.last_raw: list[RawChunk] = []

    def extract_chunks(self, text: str) -> list[RawChunk]:
        raise NotImplementedError

    def extract_labeled_chunks(self, text: str, labels: Any, examples: Any = ()) -> list[RawChunk]:
        self.last_raw = self.inner.extract_labeled_chunks(text, labels, examples)
        return self.last_raw


def generate(args: argparse.Namespace) -> None:
    from llama_cpp import Llama

    from chunklabel.llm.backend import ClientBackend
    from chunklabel.llm.client import LlamaCppClient

    docs = fid.load_pubmed(args.data, args.n, args.seed)
    pool = fid.load_pubmed(args.train, args.pool, args.seed)
    llm = Llama(model_path=str(args.model), n_ctx=args.n_ctx, n_gpu_layers=-1, verbose=False)
    backend = RecordingBackend(ClientBackend(LlamaCppClient(llm)))
    done = {(r["model"], r["selection"], r["n_examples"], r["doc"])
            for r in fid.read_records(args.out)}

    for config in args.configs.split(","):
        selection, n = config.split(":")
        seg = FewShotPromptSegmenter(
            n_examples=int(n), example_selection=cast(Literal["random", "similar"], selection),
            random_state=args.seed, backend=backend, on_align_error="skip")
        seg.fit([d.text for d in pool], [gold_segmentation(d) for d in pool])
        chosen = seg.select_examples([d.text for d in docs])
        for j, (doc, examples) in enumerate(zip(docs, chosen)):
            key = (args.name, selection, int(n), doc.pmid)
            if key in done:
                continue
            t0 = time.time()
            rec: dict[str, Any] = {"model": args.name, "selection": selection,
                                   "n_examples": int(n), "doc": doc.pmid,
                                   "examples": [pool[k].pmid for k in examples]}
            try:
                [rec["segments"]] = seg.predict([doc.text])
                rec["raw"] = [[c.category, c.quote] for c in backend.last_raw]
            except Exception as e:  # noqa: BLE001 - record and continue
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            rec["seconds"] = round(time.time() - t0, 1)
            with args.out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  {args.name} {selection}:{n} {j + 1}/{len(docs)} {rec['seconds']}s "
                  f"{rec.get('error', 'ok')[:60]}", flush=True)


def report(args: argparse.Namespace) -> None:
    docs = {d.pmid: d for d in fid.load_pubmed(args.data, args.n, args.seed)}
    records = [r for r in fid.read_records(args.out) if r["doc"] in docs]
    rows: list[dict[str, Any]] = []
    for model, selection, n in sorted({(r["model"], r["selection"], r["n_examples"])
                                       for r in records}):
        group = [r for r in records
                 if (r["model"], r["selection"], r["n_examples"]) == (model, selection, n)]
        ok = [r for r in group if "error" not in r]
        scores, accs, n_seg, n_lab, gaps, exact, quotes, secs = [], [], [], [], [], 0, 0, []
        for r in ok:
            doc = docs[r["doc"]]
            pred = [Chunk(category=lab, quote=doc.text[s:e], start=s, end=e)
                    for s, e, lab in r["segments"]]
            ref = agr.reference_chunks(doc)
            scores.append(chunk_agreement(pred, ref, doc.text, by="category"))
            accs.append(agr.label_accuracy(pred, ref, doc.text))
            n_seg.append(len(pred))
            n_lab.append(len({c.category for c in pred}))
            rep = fidelity([RawChunk(c, q) for c, q in r["raw"]], doc.text)
            gaps.append(rep.gap_coverage)
            exact += rep.n_exact
            quotes += rep.n_quotes
            secs.append(r["seconds"])
        row = {"model": model, "selection": selection, "n_examples": n, "docs": len(ok),
               "errors": len(group) - len(ok),
               "segments_per_doc": float(np.mean(n_seg)), "labels_per_doc": float(np.mean(n_lab)),
               **agr.mean_agreement(scores), "label_accuracy": float(np.mean(accs)),
               "verbatim": exact / quotes, "gap_coverage": float(np.mean(gaps)),
               "seconds_per_doc": float(np.mean(secs))}
        rows.append(row)
        print(f"{model:12s} {selection:8s} n={n} docs={row['docs']:2d} err={row['errors']} "
              f"seg/doc={row['segments_per_doc']:4.1f} hom={row['homogeneity']:.3f} "
              f"com={row['completeness']:.3f} ARI={row['ari']:.3f} "
              f"acc={row['label_accuracy']:.3f} verbatim={row['verbatim']:.3f} "
              f"gap={row['gap_coverage']:.3f} {row['seconds_per_doc']:.0f}s")
    args.results.write_text(json.dumps(
        {"n_abstracts": len(docs), "seed": args.seed, "pool": args.pool, "rows": rows},
        indent=2))
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
        p.add_argument("--pool", type=int, default=200, help="train abstracts in the pool")
        if name == "generate":
            p.add_argument("--train", type=Path, required=True)
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--name", required=True)
            p.add_argument("--configs", default="random:3,similar:3")
            p.add_argument("--n-ctx", type=int, default=8192)
        else:
            p.add_argument("--results", type=Path, default=HERE / "results.json")
    args = ap.parse_args()
    generate(args) if args.cmd == "generate" else report(args)


if __name__ == "__main__":
    main()
