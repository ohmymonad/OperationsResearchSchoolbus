# Architecture decision records

One file per decision that is load-bearing, already taken, and expensive to re-open. They exist
so a later review — human or otherwise — can tell "this was considered and settled" from "nobody
thought about this yet".

Format: **Context** (the forces), **Decision** (what was chosen), **Consequences** (what it costs
and what it buys). Status is `accepted` unless stated otherwise. Domain terms are as defined in
[`CONTEXT.md`](../../CONTEXT.md).

| # | Decision | Status |
|---|---|---|
| [0001](0001-search-over-served-stop-sets.md) | The served set is the only decision; an optimum is a minimal cover | accepted |
| [0002](0002-no-solver-library.md) | No solver library anywhere, including the exact reference | accepted |
| [0003](0003-node-indexing.md) | Stops are nodes `0..\|N\|-1`, the school is node `\|N\|` | accepted |
| [0004](0004-three-methods-three-families.md) | Three methods from three different families of the syllabus | accepted |
| [0005](0005-equalise-wall-clock.md) | Budgets equalise wall-clock time, not iterations | accepted |
| [0006](0006-three-options-by-construction.md) | The three-option requirement holds by construction, not by filtering | accepted |
| [0007](0007-working-solution-carries-its-tour.md) | A working solution carries its tour; moves edit it in place | accepted |
| [0008](0008-gaps-carry-their-reference.md) | A gap carries the kind of reference it was measured against | accepted |
| [0009](0009-disjoint-branching-no-dedup.md) | Minimal covers are enumerated by disjoint branching, with no `seen` set | accepted |
| [0010](0010-exact-search-refuses-to-degrade.md) | The exact search refuses oversized instances rather than falling back | accepted |
| [0011](0011-h3-evaluates-only-feasible-chromosomes.md) | H3 repairs every chromosome; the planned penalty term was dropped | accepted |
| [0012](0012-h3-generation-loop-ignores-the-budget.md) | H3's generation loop never reads the clock, and polishes at capture | accepted |
| [0013](0013-vectorised-two-opt.md) | 2-opt scans vectorised sweeps rather than keeping don't-look bits | accepted |
| [0014](0014-stdlib-csv-not-pandas.md) | Results are written with the stdlib `csv` module | accepted |

## Not recorded here

The **mandatory test-driven working method** is `PLAN.md` §0, not an ADR — it governs how every
line of this repository is written, which is more than a single decision.

The deepening candidates raised by the architecture review of 2026-08-26 (`review reports/`) were
vetted against the code and folded into **`PLAN.md` §5a**, which is where Phase 5's shape is
settled. They are plan, not yet decision: each becomes an ADR here once it is built and green.
The one exception is the review's `TourCache` candidate, which §5a step 0 answers by narrowing
the claim rather than widening the implementation — the reasoning is in [ADR-0007](0007-working-solution-carries-its-tour.md).
