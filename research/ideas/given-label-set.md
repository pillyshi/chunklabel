# Given label set: closed-vocabulary segmentation and labeling

**Status**: Draft

## Motivation

chunklabel currently lets the LLM name every category freely. Labels then differ
from document to document, so results cannot be compared or aggregated across a
corpus. Most practical users (and every gold-labeled benchmark) need a fixed label
set. The evaluation decision of 2026-10-10 already reserved this as a future
"given label set" feature (`research/ideas/evaluation-suite.md`).

With a fixed label set, chunklabel solves exactly the problem of semi-Markov models,
without training: partition the text into non-overlapping, fully covering segments
and give each one a label from a closed set.

## Evidence

- `catalog.yaml` [sarawagi-2004]: defines a segmentation as segments
  `<t_j, u_j, y_j>` with `y_j` from a fixed set `Y`, non-overlapping and fully
  covering. This is chunklabel's problem with a closed label set (see
  `research/notes/sarawagi-2004.md`).
- `catalog.yaml` [kong-2015]: same problem with a neural scorer; calls it "joint
  segmentation and labeling". BIO tagging is described as a reduction of it.
- `benchmarks/label_agreement/README.md`: on PubMed RCT, free labels are too fine
  (9–10 per abstract vs 4 roles), which caps category-level ARI at 0.52–0.54. A given
  label set is the first lever listed there.
- `catalog.yaml` [dernoncourt-2017]: PubMed RCT has a small closed role set, so the
  existing benchmark applies as is.

## Proposed Scope

1. Optional label set as a constructor parameter of `ZeroShotPromptSegmenter` (see
   `sklearn-estimator-api.md`), e.g.
   `ZeroShotPromptSegmenter(labels=["background", "objective", "methods", "results", "conclusions"])`.
   Optional per-label descriptions. `ChunkLabeler` stays as it is.
2. When set, the split prompts list the allowed labels and ask the LLM to use only
   those. Structured output restricts `category` to an enum where the backend supports
   it (llama.cpp grammar, OpenAI JSON schema).
3. Labels outside the set (if the backend cannot enforce the enum) are mapped to the
   closest allowed label or rejected, controlled by a parameter.
4. Without a label set, `ZeroShotPromptSegmenter` uses free labels, and `ChunkLabeler`
   is unchanged.

## Acceptance Criteria

- With `labels` set, every returned `Chunk.category` is in the set.
- `ChunkLabeler` outputs are identical to the current version (same prompts).
- `benchmarks/label_agreement` run with the five PubMed roles reports category-level
  ARI/V-measure and fidelity, next to the current free-label numbers.

## Out Of Scope

- Inducing the label set automatically (`label-schema-induction.md`).
- Training or fine-tuning any model.
- Dynamic-programming decoding (`segment-dp-decoding.md`).

## Open Questions

- Does restricting labels change boundary placement (fewer, longer chunks), or only
  names? The benchmark should report chunks/doc separately from ARI.
- Should an "other" label always be allowed, as `O` in semi-CRFs for NER?
- Can small local models (Qwen2.5-7B, Gemma 4 E4B) follow a closed set without
  grammar enforcement?
