# ADR-0013 — 2-opt scans vectorised sweeps rather than keeping don't-look bits

Status: accepted — deviates from `PLAN.md` §2 as originally approved

## Context

2-opt runs inside every method's inner loop, so its cost matters. The textbook acceleration is
*don't-look bits*: mark a city as exhausted when no improving move involves it, and skip it until
a neighbour changes.

The bits are conventionally held per city. Every 2-opt move reverses a segment, which renumbers
positions — so keeping the bits correct under reversal needs a position-lookup table maintained
alongside the tour. That table is a classic source of subtle, silent bugs: when it drifts, the
search stops examining moves it should examine and simply returns worse tours.

## Decision

For each candidate `i`, evaluate all partners `j > i` in one vectorised numpy expression and take
the best improving move. Sweep repeatedly until a full pass finds nothing. No don't-look bits, no
position table.

## Consequences

- Most of the speed-up, at the tour sizes this project reaches (≤ ~150 stops), with far less that
  can go wrong.
- The implementation is short enough to read against the definition of the move, which matters for
  a piece of code every result depends on.
- Asymptotically worse than a properly-maintained don't-look scheme; revisit only if profiling
  shows 2-opt dominating the metaheuristics' runtime, and note that any such change alters results
  at a fixed time budget (ADR-0005).
- Or-opt stays a plain Python loop: it relocates short segments, has no vectorisable inner scan of
  the same shape, and runs only on the polish path.
