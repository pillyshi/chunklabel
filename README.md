# chunklabel

A Python library for splitting text into categorized chunks using an LLM.

## Overview

chunklabel segments text into semantically coherent spans, assigning a free-form category to each. Categories are named by the LLM without a predefined schema. Each chunk's quote is a verbatim excerpt from the source text, aligned back to the original after LLM output.

```python
from chunklabel import Seam

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
pip install chunklabel
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

## Configuration

```python
seam = Seam(
    model="gpt-4o",          # LLM model to use
    fuzzy_threshold=80,      # Match threshold for rapidfuzz alignment (0–100)
)
```

## Using local LLMs

chunklabel uses LangChain's `BaseChatModel` interface internally, so any compatible model can be passed via the `llm` parameter.

**Ollama**

```python
from langchain_ollama import ChatOllama
from chunklabel import Seam

seam = Seam(llm=ChatOllama(model="llama3"))
```

**llama.cpp (OpenAI-compatible server)**

```python
from langchain_openai import ChatOpenAI
from chunklabel import Seam

seam = Seam(llm=ChatOpenAI(
    model="llama3",
    base_url="http://localhost:8080/v1",
    api_key="not-used",
))
```

Note: local models must support structured output (JSON mode). If `with_structured_output` is not reliable, wrap the model with a JSON-enforcing layer before passing it in.

## Downstream use cases

The `Chunk` list produced by chunklabel is designed as input for further analysis:

- **NLI**: score the relationship between hypotheses and chunk categories
- **NER**: analyze co-occurrence between entity labels and categories
- **Relation extraction**: map entity-pair relations to chunk categories
- **Conditional generation**: use category as a conditioning signal for language models

## License

MIT
