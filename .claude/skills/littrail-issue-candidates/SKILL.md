---
name: littrail-issue-candidates
description: >
  Use when turning paper notes and findings into issue candidates stored in
  research/ideas/. Does not create GitHub issues automatically.
---

# littrail-issue-candidates

Turns verified paper notes and findings into structured issue candidates in
`research/ideas/`. Run after `littrail-literature-work` has produced verified
notes. GitHub issue creation is a deliberate human step that happens after
candidates are reviewed — not an automatic output of this skill.

## Guardrails

- Do not create GitHub issues automatically. Write candidates to `research/ideas/` only.
- Every candidate must cite specific evidence from `research/notes/` or `research/catalog.yaml`.
- Do not expand scope beyond what the notes support. Open Questions is the right place for uncertain areas.
- Do not create candidates from unverified papers (papers not yet in `catalog.yaml`).

## Workflow

1. Read relevant `research/notes/<key>.md` files.
2. Identify a concrete, scoped idea supported by the notes.
3. Create `research/ideas/<slug>.md` using the `issue-candidate.md` template:
   - **Motivation** — the problem or opportunity the paper reveals.
   - **Evidence** — cite specific notes and catalog keys (not raw search results).
   - **Proposed Scope** — narrow and actionable; prefer small over broad.
   - **Acceptance Criteria** — observable outcomes.
   - **Out Of Scope** — explicitly list what this candidate does not cover.
   - **Open Questions** — unresolved uncertainties that need investigation before promotion.
4. Leave the candidate as `Status: Draft` until a human reviews and promotes it.

## Promotion to GitHub issue

A human decides when a candidate is ready to become a GitHub issue. The
candidate file in `research/ideas/` stays as the source of truth. Do not
create or close GitHub issues on behalf of the user without explicit
instruction.

## Output

`research/ideas/<slug>.md` with `Status: Draft`. Nothing else.
