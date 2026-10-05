# ADR-0010 — The exact search refuses oversized instances rather than falling back

Status: accepted

## Context

`solve_exact` is the ground truth behind every Tier 1 optimality gap. Its cost is dominated by the
number of minimal covers, which grows far faster than `|N|` does — and counter-intuitively grows
with *looser* constraints: at `|N| = 16`, ten children yield 37 minimal covers and sixty children
yield 216.

A common convenience is to fall back to a heuristic route when an instance is too large. Here that
would silently replace a proven optimum with a guess, and every gap computed against it would be
wrong in a way no test downstream could detect.

## Decision

Two caps, and no fallback:

- `HELD_KARP_MAX_STOPS = 18` — the routing limit, where `2^k · k²` stops being affordable.
- `EXACT_MAX_STOPS = 16` — the enumeration guardrail, a conservative default against an accidental
  hour of CPU. It sits *below* the routing limit, so no cover this search produces can be too
  large to route exactly.

`solve_exact` raises `ExactError` when an instance exceeds `max_stops`, and raises again if a
caller tries to raise `max_stops` past the Held-Karp cap.

## Consequences

- A Tier 1 result is always a proven optimum. There is no configuration in which it is not.
- Tier 1 covers a narrow band of instance sizes, and the report says so and shows the measured
  growth (0.10 s at `|N| = 16`, 0.18 s at `|N| = 18`) rather than asserting it.
- `EXACT_MAX_STOPS` is a policy default and may be raised to 18 deliberately; `HELD_KARP_MAX_STOPS`
  is an algorithmic limit and may not.
