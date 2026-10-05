# ADR-0007 — A working solution carries its tour; moves edit it in place

Status: accepted

## Context

A served set can always be routed on demand — `route(served, ...)` runs nearest-neighbour plus
local search and returns a tour. Keeping only the set, and re-routing whenever a length is needed,
would be the simpler representation.

It also breaks the local search. Nearest-neighbour on a smaller set is not a refinement of
nearest-neighbour on the larger one: dropping a redundant stop and re-routing can produce a
*longer* tour than the original had, even though removing a node from a fixed tour is
non-increasing by the triangle inequality. A descent built on re-routing is therefore not monotone
and can loop.

## Decision

`Working` carries the served set (as a `CoverState`), its tour, and that tour's length, and keeps
all three in step. `insert` splices a stop in at its cheapest position and adds the delta;
`remove` cuts one out and subtracts. Local search edits the carried tour.

H3 is the exception that proves the rule: its fitness comes from a `TourCache` keyed on the served
set, because a population revisits chromosomes and nothing in a generational algorithm needs
monotonicity under node deletion.

## Consequences

- Every improvement move is provably non-worsening, so VND terminates.
- Move evaluation is `O(|tour|)` rather than a full routing call, which is what makes H2's
  thousand-move budget possible.
- The invariant "length matches tour" is now something code can break; it is asserted in tests and
  re-established by `reoptimise`.
- `Working` is defined in `heuristics/grasp.py` and imported by H2 and H3. That location is an
  accident of build order, not a decision — see the architecture review of 2026-08-26.
