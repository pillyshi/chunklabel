# seam

A Python library for splitting text into categorized chunks using an LLM.

## Overview

seam segments text into semantically coherent spans, assigning a free-form category to each. Categories are named by the LLM without a predefined schema. Each chunk's quote is a verbatim excerpt from the source text, aligned back to the original after LLM output.

```python
from seam import Seam

seam = Seam()
chunks = seam.split(
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
pip install seam
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

### Lenient mode

- **Gaps**: unassigned spans between chunks are filled automatically as `category="uncategorized"`
- **Overlaps**: the earlier chunk takes priority; the later chunk's start is pushed forward

## Category normalization (offline)

After processing multiple texts, category names can drift across runs. A dedicated normalization step lets the LLM consolidate them in batch.

```python
from seam import Normalizer

normalizer = Normalizer()
mapping = normalizer.build_mapping(all_chunks)
# {"kick-off": "initiation", "project start": "initiation", "blocker": "obstacle", ...}

normalized_chunks = normalizer.apply(all_chunks, mapping)
```

Normalization runs offline over the full category inventory, so the LLM can make globally consistent decisions rather than local ones.

## Configuration

```python
seam = Seam(
    model="gpt-4o",          # LLM model to use
    fuzzy_threshold=80,      # Match threshold for rapidfuzz alignment (0–100)
)
```

## Downstream use cases

The `Chunk` list produced by seam is designed as input for further analysis:

- **NLI**: score the relationship between hypotheses and chunk categories
- **NER**: analyze co-occurrence between entity labels and categories
- **Relation extraction**: map entity-pair relations to chunk categories
- **Conditional generation**: use category as a conditioning signal for language models

## License

MIT
