# chunklabel

A Python library for splitting text into categorized chunks using an LLM.

## Overview

chunklabel segments text into semantically coherent spans, assigning a free-form category to each. Categories are named by the LLM without a predefined schema. Each chunk's quote is a verbatim excerpt from the source text, aligned back to the original after LLM output.

```python
from chunklabel import ChunkLabeler

labeler = ChunkLabeler()
chunks = labeler.split(
    "The project kicked off in January with a small team. "
    "Budget constraints forced a scope reduction in March. "
    "Despite the setbacks, the product launched successfully in June."
)

# [
#   Chunk(category="initiation", quote="The project kicked off in January with a small team", start=0,   end=51),
#   Chunk(category="obstacle",   quote="Budget constraints forced a scope reduction in March", start=53,  end=104),
#   Chunk(category="outcome",    quote="the product launched successfully in June", start=120, end=160),
# ]
```

## Installation

```bash
pip install chunklabel
```

For in-process inference with llama.cpp:

```bash
pip install "chunklabel[llamacpp]"
```

## Data structures

The LLM returns raw chunks without span information. Alignment is performed as a separate step, producing the final `Chunk` with character-level positions.

```python
# Intermediate: LLM output
@dataclass
class RawChunk:
    category: str   # Free-form category name assigned by the LLM
    quote: str      # Verbatim excerpt (may contain minor transcription noise)

# Final: after alignment
@dataclass
class Chunk:
    category: str   # Same as RawChunk
    quote: str      # Excerpt aligned to source text
    start: int      # Start index in source text
    end: int        # End index in source text
```

## Pipeline

```
Input text
     │
     ▼
LLM  →  [{category, quote}, ...]   (RawChunk list)
     │
     ▼
rapidfuzz alignment  →  (start, end) resolved per chunk
     │
     ▼
Span post-processing  (lenient mode)
     │  gap-filling / overlap resolution
     ▼
Chunk list
```

### Alignment

Each quote (with surrounding whitespace stripped) is located in the source text:

1. **Exact match**, searching forward from the previous quote.
2. **Exact match anywhere**, for a verbatim quote the LLM emitted out of order.
3. **Fuzzy match**: the best-matching substring (`rapidfuzz.fuzz.partial_ratio_alignment`), accepted if its score is at least `fuzzy_threshold`. Quotes shorter than 16 characters must match exactly, since short phrases produce spurious fuzzy matches.

### Lenient mode

- **Gaps**: the prompt asks the LLM to cover the whole text. Any part it skips, or whose quote fails to align, is filled automatically as `category="uncategorized"`. Check how often this happens with `split_with_report` (`gap_coverage`).
- **Overlaps**: the earlier chunk takes priority; the later chunk's start is pushed forward

## Category normalization (offline)

After processing multiple texts, category names can drift across runs. A dedicated normalization step lets the LLM consolidate them in batch.

```python
from chunklabel import Normalizer

normalizer = Normalizer()
normalizer.build_mapping(all_chunks)
# {"kick-off": "initiation", "project start": "initiation", "blocker": "obstacle", ...}

normalized_chunks = normalizer.apply(all_chunks)
```

The mapping is stored internally after `build_mapping`, so it can be passed to `apply` implicitly. To reuse the mapping across runs without calling the LLM again:

```python
# Save after building
normalizer.save("mapping.json")

# Restore later
normalizer = Normalizer.load("mapping.json")
normalized_chunks = normalizer.apply(all_chunks)
```

Normalization runs offline over the full category inventory, so the LLM can make globally consistent decisions rather than local ones.

## Segmenter estimator (experimental)

`ZeroShotPromptSegmenter` is a scikit-learn style estimator that follows `KMeans`. Prompts contain no examples (zero-shot). Each segmentation is a list of `(start, end, label)` triples that cover the text without overlapping; `text[start:end]` is the segment's text.

**Inducing a label set.** `fit(X)` splits every text with free labels, as `ChunkLabeler` does, then merges the label names over all of `X` with `Normalizer`. `fit` processes every text it gets, so pass it a sample of the size you want.

```python
from chunklabel import ZeroShotPromptSegmenter

segmenter = ZeroShotPromptSegmenter(client="gpt-4o")
segmentations = segmenter.fit_predict(texts)  # list[list[tuple[int, int, str]]]
segmenter.labels_    # the label set found in texts
segmenter.mapping_   # free label -> label in labels_

new_segmentations = segmenter.predict(new_texts)  # only labels from labels_
```

`fit_predict` makes one LLM pass over `texts`. `predict` prompts again with the label set, so calling it on the training texts can give different boundaries than `fit_predict`.

**A given label set.** Pass `labels` to use your own labels (or `labels_` after editing it by hand). `predict` then works without `fit`.

```python
segmenter = ZeroShotPromptSegmenter(
    labels=["background", "methods", "results", "conclusions"],  # or {label: description}
)
segmentations = segmenter.predict(texts)
```

- Where the backend supports it (OpenAI structured outputs, llama.cpp), the JSON schema restricts labels to the label set. Any other label is mapped to the closest allowed one (`on_unknown_label="closest"`), or dropped / rejected (`"drop"` / `"raise"`).
- Text that no aligned quote covers joins the preceding segment, so every label comes from the label set. Check how much text that was with `predict_with_report` (see "Checking fidelity" below).
- `get_params` / `set_params` follow scikit-learn conventions, so `sklearn.base.clone` works. `ChunkLabeler` is unchanged.
- Calling `predict` before `fit`, with no `labels`, raises `chunklabel.NotFittedError` (a `ValueError` and `AttributeError`, like scikit-learn's).

**Learning from examples.** `FewShotPromptSegmenter` takes labeled texts. `y` holds one segmentation per text, or `None` for an unlabeled text (accepted, not used yet). `fit` makes no LLM call: it keeps the labeled texts as an example pool. `predict` shows `n_examples` of them to the LLM, as earlier turns of the conversation, and allows only the labels that occur in `y`.

```python
from chunklabel import FewShotPromptSegmenter

segmenter = FewShotPromptSegmenter(n_examples=3, example_selection="similar")
segmenter.fit(train_texts, train_segmentations)  # [[(start, end, label), ...], ...]
segmentations = segmenter.predict(texts)
```

- `example_selection="random"` (default) uses the same examples, drawn with `random_state`, for every text. `"similar"` picks, per text, the most similar pool texts; the most similar comes last. Similarity is TF-IDF over character 3-grams by default, which needs no extra dependency and works for text without spaces. Pass `embedding=` (a function from a list of texts to a list of vectors, such as a sentence-transformers model's `encode`) to use an embedding model instead.
- A pool text identical to the input is never used as its example.
- Each example costs about twice its length in prompt tokens (its text plus the expected answer), so lower `n_examples` for long texts. Nothing is truncated automatically.

## Configuration

```python
labeler = ChunkLabeler(
    client="gpt-4o",     # model name string, or a BaseLLMClient instance
    fuzzy_threshold=85,  # match threshold for rapidfuzz alignment (0–100)
)
```

## Checking fidelity on your model and data

How faithfully the LLM quotes the source depends on the model and the text. Before
relying on a setup, run `split_with_report` on a few representative texts:

```python
from chunklabel import ChunkLabeler

labeler = ChunkLabeler(client=my_client)
for text in sample_texts:
    chunks, report = labeler.split_with_report(text)
    print(
        f"verbatim {report.exact_rate:.0%}  fuzzy {report.fuzzy_rate:.0%}  "
        f"unaligned {report.unaligned_rate:.0%}  not quoted {report.gap_coverage:.0%}"
    )
```

| field | meaning |
|---|---|
| `exact_rate` | quotes found verbatim in the source |
| `fuzzy_rate` | quotes that differ from the source but matched at or above `fuzzy_threshold` |
| `unaligned_rate` | quotes that could not be matched (dropped; their text becomes `uncategorized`) |
| `gap_coverage` | share of the source's non-whitespace characters not covered by any quote |
| `fuzzy_scores` | match score of every non-verbatim quote, to judge `fuzzy_threshold` |

`split_with_report` returns the same chunks as `split`, but never raises on alignment
errors: unaligned quotes are counted in the report instead. For a single text you can
also call `chunklabel.eval.fidelity(raw_chunks, text)` directly.

The segmenter estimators have `predict_with_report`, which returns the same report per
text. This matters even more there: they join unquoted text to the preceding segment, so
a segmentation always looks complete, and only `gap_coverage` shows how much the LLM
skipped. `ZeroShotPromptSegmenter.fit` also keeps the reports of its own pass over the
training texts in `fit_reports_`.

```python
segmentations, reports = segmenter.predict_with_report(texts)
skipped = [i for i, r in enumerate(reports) if r.gap_coverage > 0.2]
```

As a reference point, on 40 PubMed abstracts Gemma 4 E4B quoted 100% verbatim with
almost no gaps. Qwen2.5-7B quoted 99% verbatim but left 5–7% of the text unquoted
(`benchmarks/fidelity`).

## Using local LLMs

**llama.cpp (in-process)**

```python
from llama_cpp import Llama
from chunklabel import ChunkLabeler
from chunklabel.llm import LlamaCppClient

client = LlamaCppClient(Llama(model_path="path/to/model.gguf", n_ctx=4096))
labeler = ChunkLabeler(client=client)
```

**OpenAI-compatible server** (e.g. llama.cpp server, Ollama)

Set `OPENAI_BASE_URL` before constructing the client:

```bash
OPENAI_BASE_URL=http://localhost:8080/v1 python your_script.py
```

```python
from chunklabel import ChunkLabeler
from chunklabel.llm import OpenAIClient

labeler = ChunkLabeler(client=OpenAIClient(model="llama3", api_key="not-used"))
```

Note: local models must support JSON-mode structured output.

## Downstream use cases

The `Chunk` list produced by chunklabel is designed as input for further analysis:

- **NLI**: score the relationship between hypotheses and chunk categories
- **NER**: analyze co-occurrence between entity labels and categories
- **Relation extraction**: map entity-pair relations to chunk categories
- **Conditional generation**: use category as a conditioning signal for language models

## License

MIT
