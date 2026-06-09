# RAG adapter: ChunkLabelTextSplitter

**Status**: Draft

## Motivation

chunklabel produces semantically coherent chunks, but users who build RAG
pipelines with LangChain or LlamaIndex cannot use it as a drop-in replacement
for their existing text splitters without writing glue code. A thin adapter
class would make adoption frictionless.

## Evidence

- `catalog.yaml` [gao-2023]: The RAG survey documents that chunk granularity
  directly impacts retrieval precision. Fixed-length splitting ignores semantic
  boundaries and is cited as a common source of retrieval error.

## Proposed Scope

Add a `chunklabel.integrations.langchain` submodule with a single class:

```python
from chunklabel.integrations.langchain import ChunkLabelTextSplitter

splitter = ChunkLabelTextSplitter(labeler=ChunkLabeler())
docs = splitter.split_documents(documents)  # returns List[Document]
```

The class wraps `ChunkLabeler.split()` and maps each `Chunk` to a LangChain
`Document` with `metadata={"category": chunk.category, "start": chunk.start,
"end": chunk.end}`.

## Acceptance Criteria

- `ChunkLabelTextSplitter` implements the LangChain `TextSplitter` interface.
- `split_documents` preserves all original `Document` metadata and appends
  chunk-level metadata.
- The submodule is importable only when `langchain` is installed; a clear
  `ImportError` is raised otherwise.
- At least one integration test runs against a real `ChunkLabeler` instance.

## Out Of Scope

- A LlamaIndex adapter (separate candidate if the LangChain one lands well).
- Async / streaming variants.
- Automatic chunklabel configuration based on LangChain settings.

## Open Questions

- Should this live in the main `chunklabel` repo or a separate
  `chunklabel-langchain` package to avoid pulling in LangChain as a
  dependency?
- LangChain's `TextSplitter` interface has changed across versions — which
  minimum version should we target?
