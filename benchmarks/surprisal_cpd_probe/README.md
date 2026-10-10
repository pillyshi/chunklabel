# Surprisal / change point detection probe

**Summary:** the strong Choi results do not transfer to Wiki-50 (see the Wiki-50
section). On realistic text, neither surprisal/PMI nor embeddings beat a trivial
no-boundary baseline unless the number of boundaries is given.

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
    python benchmarks/surprisal_cpd_probe/probe.py \
    --dataset choi --data <text-segmentation>/data/choi/1 --per-group 25 --cache features.npz
# Wiki-50: add --with pandas --with pyarrow and
#   --dataset wiki50 --data <maiammar/wiki50>/data/test-00000-of-00001.parquet
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

## Wiki-50

- Data: Wiki-50 test set (Koshorek et al., 2018) as published in the
  [maiammar/wiki50](https://huggingface.co/datasets/maiammar/wiki50) HF dataset
  (50 documents, 3070 sentences, 6.7 boundaries per document, mean section length 8
  sentences, 15% of sections ≤ 2 sentences). The text is pre-cleaned: headings
  removed, some punctuation such as hyphens stripped. The uploader is not the
  original author. Sentence and section counts are consistent across all 50
  documents.
- Same models and code (`--dataset wiki50`); features took 556 s on CPU. The LM head
  is applied in slices of 512 positions (identical output, verified) to handle
  documents of up to ~12k tokens.

Boundary profile (offset 0 = first sentence of a new section):

| offset | −2 | −1 | **0** | +1 | +2 |
|---|---|---|---|---|---|
| surprisal | −0.00 | −0.02 | **−0.21** | +0.01 | +0.04 |
| pmi | −0.09 | −0.02 | **−0.22** | −0.09 | −0.03 |

| method | signal | Pk ↓ | WD ↓ | predicted / doc |
|---|---|---|---|---|
| no boundaries | – | **0.402** | 0.402 | 0 |
| random, known K | – | 0.467 | 0.494 | 6.7 |
| evenly spaced, known K | – | 0.491 | 0.496 | 6.7 |
| **KCPD cosine, known K** | **embedding** | **0.369** | **0.398** | 6.7 |
| KCPD linear, known K | surprisal + pmi | 0.419 | 0.457 | 6.7 |
| peaks, known K | −pmi | 0.424 | 0.447 | 6.7 |
| peaks, known K | surprisal | 0.471 | 0.488 | 6.7 |
| KCPD cosine, penalised C = 0.1 | embedding | 0.432 | 0.472 | 3.8 |
| KCPD cosine, penalised C = 0.2 | embedding | 0.398 | 0.401 | 0.4 |
| peaks z > 1.0 | −pmi | 0.453 | 0.492 | 6.8 |
| Meta-Chunking minima rule | surprisal | 0.491 | 0.558 | – |

On Choi, the same baselines are: no boundaries 0.470, evenly spaced 0.321.

### Findings on Wiki-50

1. **The surprisal spike disappears.** At section openings within one article,
   surprisal is slightly *lower* (−0.21σ), not higher. The Choi spike was mainly a
   "new unrelated document" effect, as the Choi caveat suspected.
2. **Only embeddings with an oracle K beat the trivial baseline** (0.369 vs 0.402).
   This matches the Embed-KCPD paper's Wiki-50 numbers (Pk 0.38–0.42 across
   encoders) [jia-2026], so the pipeline reproduces the literature.
3. **Choosing the number of boundaries is the unsolved part.** Every penalised or
   thresholded variant either collapses to "almost no boundaries" (≈ trivial
   baseline) or over-segments and is worse. The Choi-tuned C (0.1) transfers badly.
4. For reference, supervised models reach Pk ≈ 0.16–0.18 on Wiki-50 (CATS,
   TextSeg, as reported in [jia-2026]). Prompt-free unsupervised signals are far
   from that.
