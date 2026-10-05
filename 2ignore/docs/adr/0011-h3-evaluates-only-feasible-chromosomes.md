# ADR-0011 — H3 repairs every chromosome; the planned penalty term was dropped

Status: accepted — deviates from `PLAN.md` §2 as originally approved

## Context

`PLAN.md` §2 specified a memetic fitness of `tour length + penalty · uncovered children`, letting
infeasible individuals compete. In implementation, every chromosome passes through `repair` —
greedily cover, then drop redundant — before it is evaluated. The penalty term was therefore
unreachable: no individual with an uncovered child ever reached the fitness function.

Two ways out: remove the repair and let selection handle feasibility, or remove the penalty term.

## Decision

Keep the repair, drop the penalty term rather than leave it as dead code.

Repair is what keeps a population method competitive on a covering problem. Almost every
recombination of two covers is infeasible — that is the normal case, not an edge case — so without
repair, selection spends its entire budget rediscovering feasibility instead of recombining good
covers.

Infeasible search is **H2's** job: Tabu Search prices uncovered children with an adaptive penalty
and deliberately crosses infeasible ground. Leaving that out of H3 keeps the two methods genuinely
distinct, which is the whole point of running three (ADR-0004).

## Consequences

- H3's population is always feasible, so its incumbent is valid at every generation.
- The report describes H3 as a repair-based memetic algorithm, and the two methods stay in
  different families rather than converging on the same search.
- Reinstating the penalty term means removing repair, which is a redesign of H3, not a parameter.
