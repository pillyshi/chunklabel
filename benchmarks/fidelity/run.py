"""Quote fidelity of chunklabel's LLM modes on PubMed 20k RCT abstracts (llama.cpp).

See research/ideas/evaluation-suite.md (layer 1). Not part of the chunklabel package.

    # 1. generate raw chunks (expensive; resumable)
    uv run --extra llamacpp python benchmarks/fidelity/run.py generate \
        --data <pubmed-rct>/PubMed_20k_RCT/test.txt --model <path.gguf> --name qwen2.5-7b \
        --out <outside the repo>/raw.jsonl

    # 2. report fidelity under different alignment settings (cheap)
    uv run python benchmarks/fidelity/run.py report \
        --data <pubmed-rct>/PubMed_20k_RCT/test.txt --out <same>/raw.jsonl

The raw outputs contain abstract text (dataset license unstated), so they are written
outside the repository. Only aggregate results are committed.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chunklabel.types import RawChunk

HERE = Path(__file__).parent


@dataclass
class Abstract:
    pmid: str
    sentences: list[str]
    labels: list[str]

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


def detokenize(s: str) -> str:
    s = s.strip()
    s = re.sub(r" ([.,;:?!%)\]}])", r"\1", s)
    s = re.sub(r"([(\[{$]) ", r"\1", s)
    s = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", s)
    return s


def load_pubmed(path: Path, n: int, seed: int) -> list[Abstract]:
    abstracts: list[Abstract] = []
    cur: Abstract | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("###"):
            cur = Abstract(line[3:].strip(), [], [])
            abstracts.append(cur)
        elif line.strip() and cur is not None:
            label, sentence = line.split("\t", 1)
            cur.labels.append(label)
            cur.sentences.append(detokenize(sentence))
    return random.Random(seed).sample(abstracts, n)


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def generate(args: argparse.Namespace) -> None:
    from llama_cpp import Llama

    from chunklabel.llm.backend import ClientBackend
    from chunklabel.llm.client import LlamaCppClient

    docs = load_pubmed(args.data, args.n, args.seed)
    done = {(r["model"], r["mode"], r["doc"]) for r in read_records(args.out)}
    llm = Llama(model_path=str(args.model), n_ctx=args.n_ctx, n_gpu_layers=-1, verbose=False)
    backend = ClientBackend(LlamaCppClient(llm))
    for mode in args.modes.split(","):
        extract = backend.extract_chunks if mode == "one_pass" else backend.extract_boundaries
        for j, doc in enumerate(docs):
            if (args.name, mode, doc.pmid) in done:
                continue
            t0 = time.time()
            rec: dict[str, Any] = {"model": args.name, "mode": mode, "doc": doc.pmid}
            try:
                rec["chunks"] = [[c.category, c.quote] for c in extract(doc.text)]
            except Exception as e:  # noqa: BLE001 - record and continue
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            rec["seconds"] = round(time.time() - t0, 1)
            with args.out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  {args.name} {mode} {j + 1}/{len(docs)} {rec['seconds']}s "
                  f"{rec.get('error', 'ok')[:60]}", flush=True)


def normalize(s: str) -> str:
    """Lowercase and keep only letters and digits, for 'cosmetic difference' checks."""
    return re.sub(r"[^0-9a-z]", "", s.lower())


def partial_align(quotes: list[str], text: str, threshold: float) -> list[tuple[float, int, int]]:
    """Candidate scorer: best-matching substring via fuzz.partial_ratio_alignment,
    searching forward from the previous accepted match like chunklabel.alignment does.
    Returns (score, start, end); start = -1 when nothing was found."""
    from rapidfuzz import fuzz

    out = []
    search_start = 0
    for q in quotes:
        idx = text.find(q, search_start)
        if idx != -1:
            out.append((100.0, idx, idx + len(q)))
            search_start = idx
            continue
        r = fuzz.partial_ratio_alignment(q, text[search_start:])
        if r is None:
            out.append((0.0, -1, -1))
            continue
        out.append((r.score, search_start + r.dest_start, search_start + r.dest_end))
        if r.score >= threshold:
            search_start += r.dest_start
    return out


def report(args: argparse.Namespace) -> None:
    import numpy as np

    from chunklabel.alignment import align_detailed
    from chunklabel.eval import fidelity

    docs = {d.pmid: d for d in load_pubmed(args.data, args.n, args.seed)}
    records = [r for r in read_records(args.out) if r["doc"] in docs]
    thresholds = [50, 60, 70, 75, 80, 85, 90, 95]
    rows: list[dict[str, Any]] = []
    inspect: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((r["model"], r["mode"]), []).append(r)

    for (model, mode), recs in sorted(groups.items()):
        n_quotes = n_exact = n_cosmetic = n_verbatim_missed = errors = 0
        accepted_current = {t: 0 for t in thresholds}
        accepted_partial = {t: 0 for t in thresholds}
        gap80: list[float] = []
        overlap80 = 0
        for r in recs:
            if "error" in r:
                errors += 1
                continue
            text = docs[r["doc"]].text
            raws = [RawChunk(c, q) for c, q in r["chunks"]]
            quotes = [q for _, q in r["chunks"]]
            rep = fidelity(raws, text, threshold=80)
            gap80.append(rep.gap_coverage)
            overlap80 += rep.overlap_chars
            n_quotes += rep.n_quotes
            n_exact += rep.n_exact
            # Actual behaviour at each threshold (acceptance changes the search position).
            for t in thresholds:
                accepted_current[t] += sum(
                    d.span is not None for d in align_detailed(raws, text, threshold=t))
                accepted_partial[t] += sum(sc >= t for sc, _, _ in partial_align(quotes, text, t))
            cur = align_detailed(raws, text, threshold=80)
            par = partial_align(quotes, text, 80)
            for q, d, (ps, ps0, ps1) in zip(quotes, cur, par):
                if d.exact:
                    continue
                n_verbatim_missed += q in text
                matched = text[ps0:ps1] if ps0 >= 0 else ""
                cosmetic = normalize(q) in normalize(text)
                n_cosmetic += cosmetic
                inspect.append({
                    "model": model, "mode": mode, "doc": r["doc"], "len": len(q),
                    "current": round(d.score, 1), "partial": round(ps, 1),
                    "cosmetic": cosmetic, "verbatim_elsewhere": q in text,
                    "quote": q, "matched": matched,
                })
        n_ok = max(n_quotes, 1)
        row = {
            "model": model, "mode": mode, "docs": len(recs), "errors": errors,
            "quotes": n_quotes, "exact_rate": n_exact / n_ok,
            "non_exact": n_quotes - n_exact, "cosmetic_non_exact": n_cosmetic,
            "verbatim_but_missed": n_verbatim_missed,
            "gap_coverage@80": float(np.mean(gap80)) if gap80 else None,
            "overlap_chars@80": overlap80,
            "accepted_current": {t: v / n_ok for t, v in accepted_current.items()},
            "accepted_partial": {t: v / n_ok for t, v in accepted_partial.items()},
        }
        rows.append(row)
        print(f"{model:12s} {mode:9s} docs={len(recs)} err={errors} quotes={n_quotes} "
              f"exact={row['exact_rate']:.3f} non-exact={row['non_exact']} "
              f"(cosmetic {n_cosmetic}, verbatim-missed {n_verbatim_missed}) gap@80={row['gap_coverage@80']:.3f} "
              f"overlap@80={overlap80}")
        print("    accepted (current ratio):  " + " ".join(
            f"{t}:{v:.3f}" for t, v in row["accepted_current"].items()))
        print("    accepted (partial_ratio):  " + " ".join(
            f"{t}:{v:.3f}" for t, v in row["accepted_partial"].items()))

    (HERE / "results.json").write_text(json.dumps(
        {"n_abstracts": len(docs), "seed": args.seed, "rows": rows}, indent=2))
    # Contains abstract text: written next to the raw outputs, outside the repository.
    inspect_path = args.out.with_name("inspect.jsonl")
    inspect_path.write_text("".join(json.dumps(x) + "\n" for x in inspect))
    print(f"wrote {HERE / 'results.json'} and {inspect_path} ({len(inspect)} non-exact quotes)")


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
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--name", required=True)
            p.add_argument("--modes", default="one_pass,two_pass")
            p.add_argument("--n-ctx", type=int, default=4096)
    args = ap.parse_args()
    generate(args) if args.cmd == "generate" else report(args)


if __name__ == "__main__":
    main()
