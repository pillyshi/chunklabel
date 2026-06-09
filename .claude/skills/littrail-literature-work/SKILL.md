---
name: littrail-literature-work
description: >
  Use when discovering related papers, adding them to the catalog, and writing
  reading notes. Uses littrail search as the discovery step, not as evidence.
---

# littrail-literature-work

Guides paper discovery, catalog maintenance, and note writing for a project's
research workflow. Search results are unverified candidates until stable
identifiers and primary sources are checked.

## Guardrails

- `littrail search` results are candidates only — not verified papers.
- Do not write `research/notes/<key>.md` before inspecting the primary source.
- Do not create GitHub issues directly from search results or notes. Move ideas to `research/ideas/` first.
- Do not mutate `catalog.yaml` by hand. Use `littrail add-paper`.

## Workflow

### 1. Discover candidates

```bash
littrail search "<project-specific query>" --json --limit 20
```

The output is a JSON array of unverified candidates. Fields include
`openalex_id`, `doi`, `title`, `authors`, `year`, and `venue`.

### 2. Select promising candidates

Review the returned works for project relevance. Treat titles and venues as
hints — they are not confirmed until verified.

### 3. Add selected works

```bash
littrail add-paper --openalex <openalex_id>
# or
littrail add-paper --doi <doi>
```

Repeat for each candidate worth tracking.

### 4. Verify and check

```bash
littrail verify
littrail check
```

These commands confirm metadata against OpenAlex and check repository
consistency. Fix any reported issues before continuing.

### 5. Write reading notes

Only after a paper passes verification and you have inspected the primary source:

```bash
# <key> is the citation key in catalog.yaml (e.g., smith2023).
# Create research/notes/<key>.md using the paper-note.md template.
```

Write notes relative to this project's goals — why it matters, what is
actionable, what the limitations are.

### 6. Move implementation ideas to research/ideas/

Promising implementation or experiment ideas belong in `research/ideas/`, not
in GitHub issues. Use the `issue-candidate.md` template and fill in Evidence
from your reading notes.

## Output

| Artifact | When to create |
|---|---|
| `research/catalog.yaml` | After `littrail add-paper` + `littrail verify` |
| `research/notes/<key>.md` | After verifying and reading the primary source |
| `research/ideas/<slug>.md` | After writing notes and identifying an actionable idea |
