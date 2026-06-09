# Prompt format ablation experiment

**Status**: Draft

## Motivation

chunklabel's output quality depends heavily on the LLM prompt, but the current
quote-first format has never been compared against alternatives in a controlled
way. Different output orderings (quote vs. category first) and schema
enforcement strategies (free-form JSON vs. function calling) may significantly
affect alignment accuracy and category consistency.

## Evidence

- `catalog.yaml` [xu-2024]: Organises generative IE along three axes — output
  format, task definition, and model strategy. The paper shows that output
  format choice is a primary driver of extraction quality and varies
  significantly by task type.

## Proposed Scope

Design and run a controlled ablation over three prompt variants, measured with
Pk and WindowDiff (see `eval-metrics-pk-windowdiff.md`):

| Variant | Description |
|---------|-------------|
| A (current) | Quote-first: LLM outputs verbatim excerpt, then category |
| B | Category-first: LLM outputs category, then verbatim excerpt |
| C | Schema-explicit: JSON Schema enforced via function calling |

Each variant is tested on the same corpus under identical conditions (model,
temperature, dataset). Report Pk and WindowDiff per variant and per model size.

## Acceptance Criteria

- All three variants are implemented as interchangeable prompt templates.
- Results are reproducible via a single script with a fixed random seed.
- A summary table of Pk / WindowDiff scores is produced as output.

## Out Of Scope

- Fine-tuning or few-shot examples (zero-shot only in this experiment).
- Evaluating models other than the default OpenAI client.
- Automated prompt selection based on results.

## Open Questions

- Which corpus and ground-truth segmentation should be used? Depends on
  `eval-metrics-pk-windowdiff.md` being resolved first.
- Does category-first prompting produce more consistent category names, or
  does it hurt alignment accuracy by separating the quote from its anchor?
