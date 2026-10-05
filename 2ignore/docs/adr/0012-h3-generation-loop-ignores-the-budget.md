# ADR-0012 — H3's generation loop never reads the clock, and polishes at capture

Status: accepted

## Context

Budgets are wall-clock (ADR-0005), so the same seed reaches a different number of generations on a
busy machine than on an idle one. If a generation's behaviour depended on the elapsed time or the
remaining budget, two runs of the same seed would draw different random streams and produce
unrelated answers — and "a longer run is never worse than a shorter one", which the convergence
plots assume, would be false.

Separately: local search polish (Or-opt) is **not order-preserving**. A raw incumbent that is
better than another can polish to a worse tour than the other would. Deferring the polish to the
end of the run therefore breaks the same guarantee.

## Decision

Two invariants in `heuristics/memetic.py`:

1. The generation body never consults `tracker.elapsed` or `max_iterations`. The budget is checked
   only at the very bottom of the loop, to decide whether to run another generation. The first *k*
   generations of a long run therefore draw exactly the random stream of a short run.
2. Every incumbent is refined and polished **at the moment it is captured** (`_refine` inside the
   loop), so recorded lengths are always polished lengths compared against polished lengths.

H1 and H2 polish only at the end, which is correct for them: their incumbent is a trajectory
state, not a member of a competing population.

## Consequences

- H3 is prefix-stable: a longer run's history begins with the shorter run's history, exactly.
- H3 pays for a polish per improvement rather than one per run. That cost is real and is the price
  of the guarantee.
- Moving the `_refine` call out of the loop would look like a harmless optimisation and would
  silently break both properties. It is called out in the module and in the handover.
