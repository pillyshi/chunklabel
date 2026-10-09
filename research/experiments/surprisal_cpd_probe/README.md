# Surprisal / change point detection probe

Feasibility probe for `research/ideas/surprisal-cpd-boundaries.md`: can prompt-free
signals place segment boundaries, and which ones?

## Setup

- Data: Choi's dataset, set `1` (groups 3-5, 6-8, 9-11, 3-11), 25 random documents
  per group (seed 0) → 100 documents, 7023 sentences. Each document concatenates the
  first n sentences of 10 random Brown-corpus documents. Taken from
  [koomri/text-segmentation](https://github.com/koomri/text-segmentation) `data/choi/1`
  and lightly detokenised; not committed here.
- LM: `Qwen/Qwen2.5-0.5B` (base), transformers, CPU, fp32. One forward pass over the
  whole document for full-context surprisal, plus one pass per sentence alone for the
  local surprisal used in PMI.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
- Metrics: sentence-level Pk / WindowDiff from `chunklabel.eval`, `k` = half the mean
  reference segment length, averaged over documents.
- Runtime: 1384 s for feature extraction (~0.2 s / sentence on CPU, dominated by the
  LM).

Reproduce:

```bash
uv run --with torch --with transformers --with sentence-transformers --with ruptures \
    python research/experiments/surprisal_cpd_probe/probe.py \
    --choi-dir <text-segmentation>/data/choi/1 --per-group 25 --cache features.npz
```

## Signals

- `surprisal` — mean token surprisal of sentence i given all preceding text
  (Meta-Chunking's PPL feature in log space) [zhao-2024].
- `pmi` — surprisal of sentence i's tokens (excluding its first) given only the
  sentence so far, minus the same given the full document: how much the preceding
  document helps [tsipidi-2024].
- `embedding` — sentence embeddings [jia-2026].

## Results

Mean z-scored value around each true boundary (offset 0 = first sentence of the new
segment):

| offset | −2 | −1 | **0** | +1 | +2 |
|---|---|---|---|---|---|
| surprisal | −0.19 | −0.23 | **+0.97** | −0.02 | −0.20 |
| pmi | +0.26 | +0.30 | **−1.11** | −0.05 | +0.22 |

The boundary shows up as a **one-sentence spike**, not a level shift.

| method | signal | Pk ↓ | WD ↓ |
|---|---|---|---|
| random (known K) | – | 0.482 | 0.506 |
| Meta-Chunking minima rule (best θ) | surprisal | 0.434 | 0.483 |
| KCPD mean-shift, known K | surprisal | 0.400 | 0.465 |
| KCPD mean-shift, known K | pmi | 0.434 | 0.502 |
| peaks, known K | surprisal | 0.300 | 0.323 |
| peaks, known K | surprisal − pmi | 0.182 | 0.202 |
| **peaks, known K** | **−pmi** | **0.145** | **0.161** |
| peaks, z > 1.0 | −pmi | 0.188 | 0.208 |
| **KCPD cosine, known K** | **embedding** | **0.036** | **0.038** |
| KCPD cosine, penalised (C = 0.1) | embedding | 0.054 | 0.055 |
| KCPD linear, known K | embedding + 0.3·pmi | 0.053 | 0.059 |

"known K" gives the true number of boundaries, which isolates signal quality from
model selection. The penalised / threshold rows pick one global C, τ or θ from a small
grid **on the evaluation documents**, so they are optimistic.

## Findings

1. **Sign:** surprisal peaks and context-PMI dips *at* the first sentence of a new
   segment. This is the opposite of the Meta-Chunking rule (cut after a PPL minimum),
   and that rule is close to random here.
2. **Detector:** because the signal is an impulse, mean-shift change point detection
   is the wrong model (Pk ≈ 0.40). Peak picking is far better (0.145 with PMI).
3. **PMI > raw surprisal** (0.145 vs 0.300). Removing sentence-intrinsic difficulty
   matters, as tsipidi-2024's PMI measure suggested.
4. **Embeddings win by a wide margin** (0.036 vs 0.145) at ~1/100 of the compute.
   Adding PMI to the embedding series did not help with this naive weighting.

## Caveats

- Choi's segments are the *opening* sentences of unrelated documents, so every
  boundary sentence is also a document opening (headline-like, introduces new
  entities). That favours a surprisal spike at offset 0 and favours embeddings
  (topics are unrelated). Real texts with related, gradual segments, and
  chunklabel's sub-topic "functional" chunks, are untested.
- One small LM, one embedder, 100 documents, no confidence intervals.
- Only the PMI with no local context was tried (sentence alone). A short local
  window (previous sentence) is untested.
