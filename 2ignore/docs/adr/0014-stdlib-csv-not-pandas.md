# ADR-0014 — Results are written with the stdlib `csv` module

Status: accepted — deviates from `PLAN.md` §4 as originally approved

## Context

The plan listed `pandas` as a dependency for the results tables. The campaign's actual access
pattern is: append one flat record per run, then read the file back once to compute gaps and draw
figures. No joins, no reshaping, no time series, no group-by that a dict cannot express.

## Decision

`csv.DictWriter` and `csv.DictReader` from the standard library. Dependencies stay `numpy`,
`matplotlib` and `pytest`.

## Consequences

- One fewer dependency in a project whose dependency list is itself an argument about rule
  compliance (ADR-0002).
- The CSV field list becomes something the project must own explicitly, since no dataframe infers
  it. That is a cost, and the architecture review of 2026-08-26 proposes a module to carry it.
- Aggregation for the figures is written by hand — a few lines of `statistics` per plot. If that
  ever exceeds a screen per figure, this ADR is worth reopening.
