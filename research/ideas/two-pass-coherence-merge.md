# Two-pass coherence: merge adjacent same-cluster chunks

**Status**: Draft

## Motivation

The existing two-pass mode splits then re-labels, but does not consolidate
chunks whose categories converge to the same semantic cluster. Adjacent chunks
that the LLM assigns to near-identical categories end up as separate spans,
inflating chunk count and reducing downstream retrieval precision.

## Evidence

- `catalog.yaml` [lo-2021]: Stacking a second Transformer pass on top of a
  pre-trained encoder enforces topic coherence and improves segmentation
  quality over single-pass models. The key insight is that local LLM decisions
  can be globally inconsistent; a second pass corrects this.
- `catalog.yaml` [riedl-2012]: TopicTiling merges adjacent segments sharing the
  same dominant LDA topic, showing that cluster-based merging is an effective
  post-processing strategy.

## Proposed Scope

Add an optional post-processing step after `split()`:

1. Embed the category strings of all chunks using a lightweight model.
2. Run agglomerative clustering with a distance threshold.
3. Merge adjacent chunks that fall in the same cluster, concatenating their
   quotes and unifying their category to the cluster label (or the most
   frequent category string in the cluster).

This should compose with the offline `Normalizer` — run normalisation first,
then merge.

```python
from chunklabel import ChunkLabeler
from chunklabel.postprocess import merge_adjacent_clusters

chunks = labeler.split(text)
chunks = merge_adjacent_clusters(chunks, threshold=0.3)
```

## Acceptance Criteria

- `merge_adjacent_clusters` only merges *adjacent* chunks in the same cluster,
  not non-adjacent ones.
- Merged chunk `start` / `end` spans cover the full original span.
- The function is a no-op when all chunks are in distinct clusters.
- Unit tests cover: no-merge case, single merge, chain merge.

## Out Of Scope

- Changing the `split()` call signature.
- Selecting the embedding model automatically based on hardware.
- Merging non-adjacent chunks.

## Open Questions

- What is a sensible default clustering threshold? Needs empirical tuning
  against real outputs.
- Should this be integrated into the two-pass `split()` path or remain a
  standalone post-processing function?
