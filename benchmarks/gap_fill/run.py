"""Gap filling (issue #6) on the outputs of benchmarks/few_shot (llama.cpp).

The first call of each abstract is replayed from the few_shot records, so only the fill
calls reach the model: this measures what ``fill_gaps`` adds, at the cost of the fill
calls alone. Both settings (``fill_gaps`` on and off) go through
``FewShotPromptSegmenter.predict_with_report``.

    # 1. generate (calls the model only for gaps; resumable)
    uv run --extra llamacpp python benchmarks/gap_fill/run.py generate \
        --data <pubmed-rct>/PubMed_20k_RCT/test.txt --train <...>/train.txt \
        --few-shot <outside the repo>/few_shot.jsonl --model <path.gguf> --name qwen2.5-7b \
        --out <outside the repo>/gap_fill.jsonl

    # 2. report (cheap)
    uv run python benchmarks/gap_fill/run.py report \
        --data <pubmed-rct>/PubMed_20k_RCT/test.txt --out <same>/gap_fill.jsonl

Outputs contain abstract text, so they stay outside the repository. Only aggregates
are committed.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from chunklabel import FewShotPromptSegmenter
from chunklabel.alignment import align_detailed
from chunklabel.eval import char_labels
from chunklabel.gapfill import find_gaps
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
few = _load("few_shot_run", ROOT / "few_shot" / "run.py")


class ReplayBackend(LLMBackend):
    """Answers the full text from a recorded output; sends anything else (gaps) on."""

    def __init__(self, inner: LLMBackend | None) -> None:
        self.inner = inner
        self.recorded: dict[str, list[RawChunk]] = {}
        self.fill_calls = 0

    def extract_chunks(self, text: str) -> list[RawChunk]:
        raise NotImplementedError

    def extract_labeled_chunks(self, text: str, labels: Any, examples: Any = ()) -> list[RawChunk]:
        if text in self.recorded:
            return self.recorded[text]
        assert self.inner is not None and not examples
        self.fill_calls += 1
        return self.inner.extract_labeled_chunks(text, labels)


def generate(args: argparse.Namespace) -> None:
    from llama_cpp import Llama

    from chunklabel.llm.backend import ClientBackend
    from chunklabel.llm.client import LlamaCppClient

    docs = {d.pmid: d for d in fid.load_pubmed(args.data, args.n, args.seed)}
    pool = fid.load_pubmed(args.train, args.pool, args.seed)
    llm = Llama(model_path=str(args.model), n_ctx=args.n_ctx, n_gpu_layers=-1, verbose=False)
    backend = ReplayBackend(ClientBackend(LlamaCppClient(llm)))
    done = {(r["selection"], r["n_examples"], r["doc"]) for r in fid.read_records(args.out)}
    records = [r for r in fid.read_records(args.few_shot)
               if r["model"] == args.name and "error" not in r and r["doc"] in docs]

    for j, r in enumerate(records):
        key = (r["selection"], r["n_examples"], r["doc"])
        if key in done:
            continue
        text = docs[r["doc"]].text
        raw = [RawChunk(c, q) for c, q in r["raw"]]
        backend.recorded = {text: raw}
        spans = [d.span for d in align_detailed(raw, text, 85)]
        rec: dict[str, Any] = {"model": args.name, "selection": r["selection"],
                               "n_examples": r["n_examples"], "doc": r["doc"],
                               "gaps": find_gaps(spans, text, 20)}
        for fill in (False, True):
            seg = FewShotPromptSegmenter(
                n_examples=r["n_examples"],
                example_selection=cast(Literal["random", "similar"], r["selection"]),
                random_state=args.seed, backend=backend, fill_gaps=fill)
            seg.fit([d.text for d in pool], [few.gold_segmentation(d) for d in pool])
            backend.fill_calls = 0
            t0 = time.time()
            [segments], [report] = seg.predict_with_report([text])
            name = "fill" if fill else "no_fill"
            rec[name] = {"segments": segments, "report": dataclasses.asdict(report),
                         "fill_calls": backend.fill_calls,
                         "seconds": round(time.time() - t0, 1)}
        with args.out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  {r['selection']}:{r['n_examples']} {j + 1}/{len(records)} "
              f"fills={rec['fill']['fill_calls']} {rec['fill']['seconds']}s", flush=True)


def report(args: argparse.Namespace) -> None:
    docs = {d.pmid: d for d in fid.load_pubmed(args.data, args.n, args.seed)}
    records = [r for r in fid.read_records(args.out) if r["doc"] in docs]
    rows: list[dict[str, Any]] = []
    for model, selection, n in sorted({(r["model"], r["selection"], r["n_examples"])
                                       for r in records}):
        group = [r for r in records
                 if (r["model"], r["selection"], r["n_examples"]) == (model, selection, n)]
        row: dict[str, Any] = {"model": model, "selection": selection, "n_examples": n,
                               "docs": len(group),
                               "docs_filled": sum(r["fill"]["fill_calls"] > 0 for r in group),
                               "fill_calls": sum(r["fill"]["fill_calls"] for r in group)}
        # Label accuracy on the gaps the first call left (the text sent to fill calls).
        on_gap = {"no_fill": [0, 0], "fill": [0, 0]}
        for name in ("no_fill", "fill"):
            accs = []
            for r in group:
                doc = docs[r["doc"]]
                pred = [Chunk(category=lab, quote=doc.text[s:e], start=s, end=e)
                        for s, e, lab in r[name]["segments"]]
                ref = agr.reference_chunks(doc)
                accs.append(agr.label_accuracy(pred, ref, doc.text))
                gap = [i for gs, ge in r["gaps"] for i in range(gs, ge)
                       if not doc.text[i].isspace()]
                p = char_labels(pred, len(doc.text))
                g = char_labels(ref, len(doc.text))
                keep = [i for i in gap if g[i] is not None]
                on_gap[name][0] += sum(str(p[i]).lower() == str(g[i]).lower() for i in keep)
                on_gap[name][1] += len(keep)
            row[f"label_accuracy_{name}"] = float(np.mean(accs))
            row[f"gap_coverage_{name}"] = float(np.mean(
                [r[name]["report"]["gap_coverage"] for r in group]))
            row[f"gap_label_accuracy_{name}"] = on_gap[name][0] / max(on_gap[name][1], 1)
        row["gap_chars"] = on_gap["fill"][1]
        row["extra_seconds_per_doc"] = float(np.mean(
            [r["fill"]["seconds"] - r["no_fill"]["seconds"] for r in group]))
        rows.append(row)
        print(f"{model:12s} {selection:8s} n={n} docs={row['docs']} "
              f"filled={row['docs_filled']} calls={row['fill_calls']} "
              f"gap={row['gap_coverage_no_fill']:.3f}->{row['gap_coverage_fill']:.3f} "
              f"gap_acc={row['gap_label_accuracy_no_fill']:.3f}->"
              f"{row['gap_label_accuracy_fill']:.3f} "
              f"acc={row['label_accuracy_no_fill']:.3f}->{row['label_accuracy_fill']:.3f} "
              f"+{row['extra_seconds_per_doc']:.1f}s")
    args.results.write_text(json.dumps(
        {"n_abstracts": len(docs), "seed": args.seed, "rows": rows}, indent=2))
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
            p.add_argument("--train", type=Path, required=True)
            p.add_argument("--pool", type=int, default=200)
            p.add_argument("--few-shot", type=Path, required=True)
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--name", required=True)
            p.add_argument("--n-ctx", type=int, default=8192)
        else:
            p.add_argument("--results", type=Path, default=HERE / "results.json")
    args = ap.parse_args()
    generate(args) if args.cmd == "generate" else report(args)


if __name__ == "__main__":
    main()
