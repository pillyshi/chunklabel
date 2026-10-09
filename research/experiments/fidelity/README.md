# Quote fidelity and the default fuzzy threshold

Layer 1 of `research/ideas/evaluation-suite.md`: how faithfully do chunklabel's LLM
quotes map back to the source, and what should `fuzzy_threshold` be?

## 1. Real LLM outputs (`run.py`)

- Data: 40 random abstracts from the PubMed 20k RCT test set (seed 0), lightly
  detokenised. Raw outputs are kept outside the repository (dataset license unstated).
  Only `results.json` (aggregates) is committed.
- Models (llama.cpp, M2 16 GB): Qwen2.5-7B-Instruct Q4_K_M (bartowski),
  Gemma 4 E4B-it QAT q4_0 (google). Library prompts unchanged; temperature 0.
- About 30–40 s per abstract per mode.

| model / mode | quotes | verbatim | non-verbatim | gap coverage @80 |
|---|---|---|---|---|
| Gemma 4 E4B, one_pass | 483 | **100%** | 0 | 0.2% |
| Gemma 4 E4B, two_pass | 482 | **100%** | 0 | 0.0% |
| Qwen2.5-7B, one_pass | 400 | 99.5% | 2 | **5.5%** |
| Qwen2.5-7B, two_pass | 488 | 99.4% | 3 | 0.0% |

The 5 non-verbatim Qwen quotes, inspected by hand:

| cause | n | current score | handled correctly at 80? |
|---|---|---|---|
| trailing space only | 2 | 99.0, 99.5 | yes, but only via fuzzy search |
| quote moved out of order (verbatim, but appears earlier in the text than the previous quote) | 1 | 42 | **no**: forward-only search misses it, so its text becomes an `uncategorized` gap |
| misspelled drug name | 1 | 93.8 | yes |
| words inserted ("non-blinded, randomised") | 1 | 88.2 | yes (arguably a paraphrase) |

Findings:

- On short, clean abstracts, both models almost never paraphrase. Real data gives
  too few non-verbatim cases to set a threshold empirically, hence the simulation
  below.
- The largest real fidelity loss is **coverage, not paraphrase**: Qwen one_pass
  leaves 5.5% of characters unquoted (skipped sentences), which `postprocess` fills
  as `uncategorized`.
- Two cheap robustness fixes are suggested by the failures: strip the quote before
  the exact search, and fall back to a global exact search when the forward search
  fails, so that a verbatim quote that is out of order is still found.

## 2. Score simulation (`threshold_sim.py`)

No LLM. Positives are real sentences (or 2–5 word phrases) of an abstract with
controlled edits, aligned against that abstract. Negatives are sentences or phrases
from a different abstract (same domain). 200 abstracts.

### The current scorer has a length-dependent ceiling

`chunklabel.alignment` scores a quote with `fuzz.ratio` against a window of
`len(quote) + 20` characters, so even a perfect window scores at most
`2L / (2L + 20)`. Measured with one deleted character:

| quote length | 20 | 30 | 40 | 60 | 80 | 120 | 200 |
|---|---|---|---|---|---|---|---|
| score | 65.5 | 74.4 | 79.6 | 85.5 | 88.8 | 92.2 | 95.2 |

At the default threshold of 80, **no non-verbatim quote shorter than ~40 characters
can ever align**, however small the error. The threshold cannot be chosen
independently of quote length.

### Sentence-level quotes (median score, by quote length)

| edit | scorer | < 40 | 40–80 | 80–160 | ≥ 160 |
|---|---|---|---|---|---|
| 1 char edit | current | 73.8 | 86.0 | 92.1 | 95.4 |
| | partial_ratio | 96.6 | 98.5 | 99.2 | 99.5 |
| drop 1 word | current | 75.6 | 86.5 | 92.5 | 95.6 |
| | partial_ratio | 82.6 | 90.9 | 95.3 | 97.4 |
| insert 3 words | current | 36.5 | 57.2 | 75.6 | 85.1 |
| | partial_ratio | 48.0 | 65.2 | 81.5 | 88.9 |
| negative (other abstract) | current | 39.3 | 44.4 | 44.1 | 42.9 |
| | partial_ratio | 45.3 | 46.3 | 44.8 | 43.1 |

Overall acceptance (all edit types pooled; no sentence-level negative is accepted at
any threshold ≥ 60 by either scorer):

| threshold | 70 | 75 | 80 | 85 | 90 |
|---|---|---|---|---|---|
| current: positives accepted | 0.970 | 0.939 | 0.885 | 0.792 | 0.616 |
| partial_ratio: positives accepted | 0.986 | 0.975 | 0.956 | 0.904 | 0.818 |

### Short phrases are where false matches appear

1 character edit (positives) vs a phrase from another abstract (negatives):

| phrase | scorer | pos @80 | neg @80 | pos @85 | neg @85 | pos @90 | neg @90 |
|---|---|---|---|---|---|---|---|
| 2 words | current | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| | partial_ratio | 0.976 | **0.061** | 0.866 | 0.008 | 0.674 | 0.003 |
| 3 words | current | 0.008 | 0.000 | 0.006 | 0.000 | 0.003 | 0.000 |
| | partial_ratio | 1.000 | 0.013 | 1.000 | 0.000 | 0.972 | 0.000 |
| 5 words | current | 0.108 | 0.000 | 0.005 | 0.000 | 0.000 | 0.000 |
| | partial_ratio | 1.000 | 0.003 | 1.000 | 0.000 | 1.000 | 0.000 |

## Conclusions

1. The current scorer conflates quote length with similarity. Its "safety" on short
   quotes is only because it can never accept them.
2. `fuzz.partial_ratio_alignment` gives length-independent scores for character-level
   noise. It also returns the matched substring's end offset, whereas the current code
   assumes `start + len(quote)`.
3. With `partial_ratio`, **85** separates positives from negatives well: ≥ 3-word
   quotes with a typo are accepted 100%, and negatives ≤ 0.8% (2-word phrases only).
   Three inserted words in a sentence (a real paraphrase) fall around the threshold.
   That seems acceptable: the threshold decides how much paraphrase to tolerate.
4. Very short quotes (≤ 2 words / ~15 characters) should require an exact match, or a
   higher threshold, because their false-match rate grows.

Caveats: PubMed abstracts only, two 4–8B models, English, and synthetic edits as a
proxy for paraphrase. Longer documents (Wiki-50 runs earlier produced long quotes)
and Japanese text are untested.
