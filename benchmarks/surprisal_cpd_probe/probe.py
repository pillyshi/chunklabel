"""Probe: do LM surprisal series carry segment-boundary signal (Choi / Wiki-50)?

See research/ideas/surprisal-cpd-boundaries.md. Not part of the chunklabel package.

Usage (dependencies are pulled in ad hoc, not added to the project):

    uv run --with torch --with transformers --with sentence-transformers --with ruptures \
        python benchmarks/surprisal_cpd_probe/probe.py \
        --dataset choi --data <path to koomri/text-segmentation>/data/choi/1 --per-group 25

    (for Wiki-50 add --with pandas --with pyarrow and use
    --dataset wiki50 --data <maiammar/wiki50>/data/test-00000-of-00001.parquet)

Per document (a concatenation of 10 Brown-corpus excerpts, one sentence per line),
three sentence-level series are computed:

- surprisal: mean token surprisal of sentence i given all preceding text
  (Meta-Chunking's PPL feature, in log space), from one causal-LM forward pass.
- pmi: mean over sentence i's tokens (excluding its first token) of
  surprisal(given only the sentence so far) - surprisal(given the full document),
  i.e. how much the preceding document helps (tsipidi-2024).
- embedding: sentence-transformer vectors (Embed-KCPD, jia-2026).

Detection methods are scored with sentence-level Pk / WindowDiff.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ruptures as rpt
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from chunklabel.eval import pk_from_boundaries, window_diff_from_boundaries

SEPARATOR = "=========="


@dataclass
class Document:
    name: str
    sentences: list[str]
    boundaries: set[int]  # sentence index where a new segment starts


@dataclass
class Features:
    surprisal: np.ndarray  # (T,)
    pmi: np.ndarray  # (T,)
    embedding: np.ndarray  # (T, d)


def detokenize(line: str) -> str:
    s = line.strip()
    s = s.replace("``", '"').replace("''", '"')
    s = re.sub(r" ([.,;:?!%)\]}])", r"\1", s)
    s = re.sub(r"([(\[{$]) ", r"\1", s)
    s = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", s)
    return s


def load_choi(root: Path, per_group: int, seed: int) -> list[Document]:
    rng = random.Random(seed)
    docs = []
    for group in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(group.glob("*.ref"), key=lambda p: int(p.stem))
        for path in rng.sample(files, min(per_group, len(files))):
            sentences: list[str] = []
            boundaries: set[int] = set()
            for line in path.read_text(encoding="latin-1").splitlines():
                if line.strip() == SEPARATOR:
                    if sentences:
                        boundaries.add(len(sentences))
                    continue
                s = detokenize(line)
                if s:
                    sentences.append(s)
            boundaries.discard(len(sentences))
            docs.append(Document(f"{group.name}/{path.stem}", sentences, boundaries))
    return docs


def load_wiki50(path: Path) -> list[Document]:
    """Wiki-50 (Koshorek et al., 2018) as the parquet in the maiammar/wiki50 HF dataset:
    sentences joined by <|end_sentence|>, chunk_end_positions = inclusive end indices."""
    import pandas as pd

    docs = []
    for row in pd.read_parquet(path).itertuples():
        sentences = [s.strip() for s in row.cleaned_text.split("<|end_sentence|>") if s.strip()]
        ends = [int(e) for e in row.chunk_end_positions]
        boundaries = {e + 1 for e in ends if e + 1 < len(sentences)}
        docs.append(Document(row.filename, sentences, boundaries))
    return docs


class SurprisalScorer:
    def __init__(self, model_name: str, device: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def _token_surprisal(self, ids: list[int]) -> np.ndarray:
        """Surprisal (nats) of ids[1:] given the preceding ids; returns len(ids) - 1 values."""
        x = torch.tensor([ids], device=self.device)
        # Project hidden states to the vocabulary in slices: full logits for a long
        # document (~12k tokens x 151k vocab) would not fit in memory.
        hidden = self.model.base_model(x).last_hidden_state[0, :-1]
        head = self.model.get_output_embeddings()
        target = x[0, 1:]
        out = [
            F.cross_entropy(head(hidden[a : a + 512]), target[a : a + 512], reduction="none")
            for a in range(0, len(target), 512)
        ]
        return torch.cat(out).cpu().numpy()

    def features(self, sentences: list[str]) -> tuple[np.ndarray, np.ndarray]:
        text = " ".join(sentences)
        starts = []
        pos = 0
        for s in sentences:
            starts.append(pos)
            pos += len(s) + 1
        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids: list[int] = enc["input_ids"]
        # Assign each token to the sentence containing its last character.
        token_sent = np.searchsorted(starts, [e - 1 for _, e in enc["offset_mapping"]], "right") - 1

        # Global: one pass over the whole document. Token 0 has no prediction.
        global_s = np.full(len(ids), np.nan)
        global_s[1:] = self._token_surprisal(ids)

        T = len(sentences)
        surprisal = np.full(T, np.nan)
        pmi = np.full(T, np.nan)
        sent_idx = [np.flatnonzero(token_sent == i) for i in range(T)]
        multi = [i for i in range(T) if len(sent_idx[i]) >= 2]
        # Local: each sentence alone; compare on tokens 1.. of the sentence.
        locals_ = self._batch_token_surprisal([[ids[j] for j in sent_idx[i]] for i in multi])
        local_of = dict(zip(multi, locals_))
        for i in range(T):
            idx = sent_idx[i]
            if len(idx) == 0:
                continue
            g = global_s[idx]
            surprisal[i] = np.nanmean(g) if not np.all(np.isnan(g)) else np.nan
            if i in local_of:
                pmi[i] = float(np.mean(local_of[i] - g[1:]))
        return _fill(surprisal), _fill(pmi)

    @torch.no_grad()
    def _batch_token_surprisal(self, seqs: list[list[int]], batch: int = 1) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        pad = self.tokenizer.pad_token_id or 0
        for b in range(0, len(seqs), batch):
            chunk = seqs[b : b + batch]
            width = max(len(s) for s in chunk)
            x = torch.full((len(chunk), width), pad, device=self.device)
            mask = torch.zeros((len(chunk), width), dtype=torch.long, device=self.device)
            for r, s in enumerate(chunk):
                x[r, : len(s)] = torch.tensor(s, device=self.device)
                mask[r, : len(s)] = 1
            logits = self.model(x, attention_mask=mask).logits[:, :-1]
            nll = F.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="none").cpu().numpy()
            out.extend(nll[r, : len(s) - 1] for r, s in enumerate(chunk))
        return out


def _fill(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x) if not np.all(np.isnan(x)) else 0.0
    return np.where(np.isnan(x), med, x)


def zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


# --- detection -------------------------------------------------------------


def known_k(signal: np.ndarray, kernel: str, k: int) -> set[int]:
    algo = rpt.KernelCPD(kernel=kernel, min_size=1).fit(signal)
    return set(algo.predict(n_bkps=k)[:-1])


def penalized(signal: np.ndarray, kernel: str, c: float) -> set[int]:
    t = len(signal)
    algo = rpt.KernelCPD(kernel=kernel, min_size=1).fit(signal)
    return set(algo.predict(pen=c * math.sqrt(t * math.log(t)))[:-1])


def minima_rule(ppl: np.ndarray, theta: float) -> set[int]:
    """Meta-Chunking PPL chunking: cut after local minima of the PPL series.

    A minimum at sentence i (both neighbours higher, one side by more than theta, or
    left higher by more than theta and right equal) ends a chunk, so the next chunk
    starts at i + 1.
    """
    out = set()
    for i in range(1, len(ppl) - 1):
        left, right = ppl[i - 1] - ppl[i], ppl[i + 1] - ppl[i]
        if (left > 0 and right > 0 and max(left, right) > theta) or (left > theta and right == 0):
            out.add(i + 1)
    return {b for b in out if 0 < b < len(ppl)}


def peaks_known_k(x: np.ndarray, k: int, radius: int = 1) -> set[int]:
    """Top-k sentences by value, with non-maximum suppression within ``radius``.

    The boundary is placed *at* the peak: the spiking sentence starts the new segment.
    """
    chosen: set[int] = set()
    for i in np.argsort(-x):
        if len(chosen) == k:
            break
        if i == 0 or any(abs(int(i) - j) <= radius for j in chosen):
            continue
        chosen.add(int(i))
    return chosen


def peaks_threshold(x: np.ndarray, tau: float, radius: int = 1) -> set[int]:
    """All sentences whose z-score exceeds ``tau``, with non-maximum suppression."""
    z = zscore(x)
    return peaks_known_k(z, int((z > tau).sum()), radius)


def spike_signals(f: Features) -> dict[str, np.ndarray]:
    s, p = zscore(f.surprisal), zscore(f.pmi)
    return {"surprisal": s, "-pmi": -p, "surprisal-pmi": s - p}


def signals(f: Features) -> dict[str, tuple[np.ndarray, str]]:
    s, p = zscore(f.surprisal)[:, None], zscore(f.pmi)[:, None]
    return {
        "surprisal": (s, "linear"),
        "pmi": (p, "linear"),
        "surprisal+pmi": (np.hstack([s, p]), "linear"),
        "embedding": (f.embedding, "cosine"),
        "embedding+pmi": (np.hstack([f.embedding, p * 0.3]), "linear"),
    }


# --- analysis --------------------------------------------------------------


def score(docs: list[Document], preds: list[set[int]]) -> tuple[float, float]:
    pks, wds = [], []
    for d, b in zip(docs, preds):
        n = len(d.sentences)
        pks.append(pk_from_boundaries(d.boundaries, b, n))
        wds.append(window_diff_from_boundaries(d.boundaries, b, n))
    return float(np.mean(pks)), float(np.mean(wds))


def boundary_profile(docs: list[Document], series: list[np.ndarray], width: int = 2) -> list[float]:
    """Mean z-scored value at offsets -width..+width from each boundary (offset 0 = first
    sentence of the new segment)."""
    acc: dict[int, list[float]] = {o: [] for o in range(-width, width + 1)}
    for d, x in zip(docs, series):
        z = zscore(x)
        for b in d.boundaries:
            for o in acc:
                if 0 <= b + o < len(z):
                    acc[o].append(float(z[b + o]))
    return [float(np.mean(acc[o])) for o in sorted(acc)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["choi", "wiki50"], default="choi")
    ap.add_argument("--data", type=Path, required=True,
                    help="choi: directory such as data/choi/1; wiki50: test parquet file")
    ap.add_argument("--per-group", type=int, default=25)
    ap.add_argument("--lm", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu or mps")
    ap.add_argument("--cache", type=Path, default=None, help="npz file to cache features")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("results.json"))
    args = ap.parse_args()

    device = args.device
    if args.dataset == "choi":
        docs = load_choi(args.data, args.per_group, args.seed)
    else:
        docs = load_wiki50(args.data)
    print(f"{len(docs)} documents, device={device}")

    scorer = SurprisalScorer(args.lm, device)
    embedder = SentenceTransformer(args.embedder, device=device)
    feats: list[Features] = []
    t0 = time.time()
    n_sent = 0
    cache: dict[str, np.ndarray] = {}
    if args.cache and args.cache.exists():
        cache = dict(np.load(args.cache))
    for j, d in enumerate(docs):
        key = d.name.replace("/", "_")
        if f"{key}_s" in cache:
            s, p, e = cache[f"{key}_s"], cache[f"{key}_p"], cache[f"{key}_e"]
        else:
            s, p = scorer.features(d.sentences)
            e = np.asarray(embedder.encode(d.sentences, normalize_embeddings=True))
            cache.update({f"{key}_s": s, f"{key}_p": p, f"{key}_e": e})
        feats.append(Features(s, p, e))
        n_sent += len(d.sentences)
        if (j + 1) % 10 == 0:
            print(f"  {j + 1}/{len(docs)} docs, {time.time() - t0:.0f}s", flush=True)
            if args.cache:
                np.savez(args.cache, **cache)
    if args.cache:
        np.savez(args.cache, **cache)
    elapsed = time.time() - t0
    print(f"features: {elapsed:.1f}s for {n_sent} sentences")

    results: dict[str, object] = {
        "n_docs": len(docs),
        "n_sentences": n_sent,
        "feature_seconds": elapsed,
        "lm": args.lm,
        "embedder": args.embedder,
    }

    # Sign of the signal around boundaries.
    profiles = {
        "surprisal": boundary_profile(docs, [f.surprisal for f in feats]),
        "pmi": boundary_profile(docs, [f.pmi for f in feats]),
    }
    results["boundary_profile_offsets_-2..+2"] = profiles
    for name, prof in profiles.items():
        print(f"profile {name:10s} " + " ".join(f"{v:+.2f}" for v in prof))

    rows: list[dict[str, object]] = []

    def add(method: str, signal: str, preds: list[set[int]]) -> None:
        pk, wd = score(docs, preds)
        rows.append({"method": method, "signal": signal, "pk": pk, "wd": wd})
        print(f"{method:28s} {signal:15s} Pk={pk:.3f} WD={wd:.3f}")

    rng = random.Random(args.seed)
    add("random (known K)", "-", [
        set(rng.sample(range(1, len(d.sentences)), len(d.boundaries))) for d in docs
    ])

    sigs = [signals(f) for f in feats]
    for name in sigs[0]:
        add("KCPD known K", name, [
            known_k(sg[name][0], sg[name][1], len(d.boundaries)) for d, sg in zip(docs, sigs)
        ])

    # Penalised: one global C per signal, best over a grid (tuned on the evaluation set,
    # so this is an optimistic number; the known-K rows are the cleaner comparison).
    grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    for name in sigs[0]:
        best = None
        for c in grid:
            preds = [penalized(sg[name][0], sg[name][1], c) for sg in sigs]
            pk, wd = score(docs, preds)
            if best is None or pk < best[1]:
                best = (c, pk, wd)
        assert best is not None
        rows.append({"method": f"KCPD pen C={best[0]}", "signal": name, "pk": best[1], "wd": best[2]})
        print(f"{'KCPD pen C=' + str(best[0]):28s} {name:15s} Pk={best[1]:.3f} WD={best[2]:.3f}")

    # Spike detection: the boundary sentence is an outlier, not a level shift.
    spikes = [spike_signals(f) for f in feats]
    for name in spikes[0]:
        add("peaks known K", name, [
            peaks_known_k(sp[name], len(d.boundaries)) for d, sp in zip(docs, spikes)
        ])
    for name in spikes[0]:
        best = None
        for tau in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            pk, wd = score(docs, [peaks_threshold(sp[name], tau) for sp in spikes])
            if best is None or pk < best[1]:
                best = (tau, pk, wd)
        assert best is not None
        rows.append({"method": f"peaks z>{best[0]}", "signal": name, "pk": best[1], "wd": best[2]})
        print(f"{'peaks z>' + str(best[0]):28s} {name:15s} Pk={best[1]:.3f} WD={best[2]:.3f}")

    # Meta-Chunking minima rule on raw sentence surprisal, theta grid.
    best = None
    for theta in [0.0, 0.1, 0.2, 0.5, 1.0]:
        pk, wd = score(docs, [minima_rule(f.surprisal, theta) for f in feats])
        if best is None or pk < best[1]:
            best = (theta, pk, wd)
    assert best is not None
    rows.append({"method": f"minima rule theta={best[0]}", "signal": "surprisal", "pk": best[1], "wd": best[2]})
    print(f"{'minima rule theta=' + str(best[0]):28s} {'surprisal':15s} Pk={best[1]:.3f} WD={best[2]:.3f}")

    results["rows"] = rows
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
