# ADR-0009 — Minimal covers are enumerated by disjoint branching, with no `seen` set

Status: accepted

## Context

The exact search needs every minimal cover exactly once. The obvious implementation — recurse over
stops, collect feasible irredundant sets, and keep a `seen` set to filter duplicates — is easy to
write and quietly quadratic in memory: the number of minimal covers grows fast (216 covers on a
16-stop, 60-child instance), and the table must hold all of them for the whole search.

## Decision

At every node the search picks the **most constrained uncovered child** — the one with the fewest
options still admissible — and branches on its options, each branch forbidding the options that
earlier branches already took.

Because the branches partition the space, every minimal cover appears exactly **once**. No `seen`
table exists. Descent stops at the first feasible set, since every strict superset of a cover
contains a redundant stop and nothing minimal lies below one.

A child with a single admissible option forces that stop rather than branching on it, and a child
with none prunes the subtree immediately.

## Consequences

- Memory is the recursion depth, not the number of covers found.
- The enumeration is a generator, so `solve_exact` can prune against a tightening incumbent as it
  goes (the MST bound).
- **A `seen` set would be a bug indicator, not an optimisation.** If duplicates ever appear, the
  branching has been broken and adding dedup would hide it. This is stated in the module docstring
  and asserted in the tests.
- Ordering is deterministic, so the exact search returns the same optimal tour every run.
