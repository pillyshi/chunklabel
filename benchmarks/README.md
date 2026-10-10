# Benchmarks

Experiment code, results, and write-ups for chunklabel. Tracked in git, but excluded from
the package build (`[tool.hatch.build] exclude` in `pyproject.toml`).

The literature trail (paper catalog, reading notes, idea candidates) lives in
`research/`. Ideas there link to the experiments here.

| directory | question | idea |
|---|---|---|
| [`surprisal_cpd_probe/`](surprisal_cpd_probe/README.md) | Can prompt-free signals (LM surprisal, context PMI, sentence embeddings + change point detection) find segment boundaries? Choi and Wiki-50. | `research/ideas/surprisal-cpd-boundaries.md` |
| [`fidelity/`](fidelity/README.md) | How faithfully do LLM quotes map back to the source, and what should the fuzzy alignment threshold be? | `research/ideas/evaluation-suite.md` |

## Conventions

- Run scripts from the repository root with `uv run`. Heavy dependencies are pulled
  in ad hoc with `--with` (see each script's docstring) and are not project
  dependencies.
- Datasets and raw model outputs stay outside the repository (licenses vary, and
  outputs can contain source text). Commit code, aggregate results (`*.json`), and the
  README with findings and caveats.
- Each README records the setup (data, models, hardware), the results, the findings,
  and the caveats, so a result can be read without rerunning it.
