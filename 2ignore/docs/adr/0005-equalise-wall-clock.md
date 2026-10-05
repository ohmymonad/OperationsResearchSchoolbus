# ADR-0005 — Budgets equalise wall-clock time, not iterations

Status: accepted

## Context

"Iteration" means something different in each method. One H1 iteration is a full construction plus
a VND descent; one H2 iteration is a single sampled move; one H3 iteration is a whole generation
of twenty individuals. Measured at a two-second budget, H2 runs ~1200 iterations, H1 ~28 and
H3 ~7.

Comparing methods at equal iteration counts would therefore compare a hundredth of one run
against a whole other run, and the resulting table would say nothing about either method.

## Decision

`Budget` is built around `max_seconds`. Iteration and stagnation limits exist as extras (and for
tests), but the campaign equalises time. Convergence plots use elapsed time on the x-axis, never
iterations.

`BudgetTracker` takes its clock as an argument, so stopping logic is tested without sleeping.

## Consequences

- Results depend on the machine they were produced on, so the report has to state it.
- Runs are not bit-reproducible at a time budget: the same seed reaches a different iteration
  count on a busier machine. Determinism tests therefore fix `max_iterations` instead, and H3
  additionally guarantees prefix-stability (ADR-0012).
- Any change that makes one method faster per iteration — a cache, a vectorisation — changes that
  method's results at fixed time. Such changes must land before a campaign starts, never between
  families of one campaign.
