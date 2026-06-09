# Boundary confidence score on Chunk

**Status**: Draft

## Motivation

Users have no signal for how confident chunklabel is about each segment
boundary. All boundaries are treated equally, even when the LLM's quoted spans
are close together or when adjacent categories are semantically similar.
Exposing a confidence score would let users filter or flag weak boundaries
without re-running the LLM.

## Evidence

- `catalog.yaml` [hearst-1994]: TextTiling detects boundaries as dips in
  lexical cohesion between adjacent text windows. The magnitude of the dip
  is a natural confidence proxy — small dips indicate uncertain boundaries.

## Proposed Scope

Add an optional `boundary_score` field to the `Chunk` dataclass:

```python
@dataclass
class Chunk:
    category: str
    quote: str
    start: int
    end: int
    boundary_score: float | None = None
```

Compute it as the cosine distance between sentence embeddings of adjacent
chunk quotes. Populate it only when the caller opts in (e.g. via a
`ChunkLabeler` flag) to avoid adding an embedding dependency to the default
path.

## Acceptance Criteria

- `boundary_score` is `None` by default.
- When enabled, the score is a float in [0, 1] where higher means a stronger
  (more confident) boundary.
- Unit tests cover the opt-in/opt-out behaviour and boundary score ordering on
  a synthetic example.

## Out Of Scope

- Automatic filtering or merging based on the score (a separate idea).
- Integration with the two-pass mode.
- Changing the existing `Chunk` serialisation format.

## Open Questions

- Which embedding model should be the default? A lightweight model
  (e.g. `all-MiniLM-L6-v2`) avoids heavy dependencies.
- Should this be a separate post-processing function rather than a dataclass
  field, to keep `Chunk` a plain data container?
