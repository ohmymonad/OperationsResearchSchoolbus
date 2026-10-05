# ADR-0008 — A gap carries the kind of reference it was measured against

Status: accepted

## Context

Two different numbers are both called "the gap" in this project:

- On small instances the true optimum is computable, so `(length - optimum) / optimum` is a real
  **optimality gap** — a proven statement.
- On everything larger, no optimum is available without a solver (ADR-0002), so quality is
  measured against the **best known solution**: the shortest tour any method at any seed found.
  That is a comparison between heuristics, and claims nothing about optimality.

Reporting the second as if it were the first would be a false claim in the report. The natural
place for that mistake is three modules away from the computation, in whoever writes the axis
title.

## Decision

`bounds.Gap` is a value carrying `value`, `reference` and `kind` (`"optimum"` or `"best-known"`),
and it renders its own label via `GAP_LABELS`. The wording of those labels is part of the
contract, not cosmetic — it is what appears on plot axes and in table headers.

Two constructors, deliberately asymmetric:

- `optimality_gap` **raises** on a length below the optimum. Beating a proven optimum is not a
  good result; it means the enumeration missed a cover, the pruning is unsound, or the routing is
  wrong. The exception fires at the moment the number is computed rather than surviving into a
  table.
- `bks_gap` **permits** negative values. Beating the incumbent best-known solution mid-campaign is
  normal, and it is the moment the registry updates.

## Consequences

- A mislabelled gap requires actively constructing the wrong `Gap`; it cannot happen by omission.
- Tier 1 and Tier 2 rows can sit in one results table without the distinction being lost.
- The two constructors cannot be unified behind a flag without discarding the asymmetry, which is
  the point of having two.
