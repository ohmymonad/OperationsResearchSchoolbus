# ADR-0002 — No solver library anywhere, including the exact reference

Status: accepted

## Context

The course specification allows data-structure and algorithm libraries but forbids handing the
whole problem to a solver; solvers are permitted only for estimating error or inside a
math-heuristic. An ILP formulation of this problem is easy to write, and calling CBC or OR-Tools
for the Tier 1 optimum would have been the shortest path to ground truth.

That would leave the project's compliance resting on an argument about what "estimating error"
covers — an argument that has to be made to a grader, in a report, after the fact.

## Decision

No mathematical-programming library is used at any point. Dependencies are `numpy`, `matplotlib`
and `pytest`, and nothing else.

This extends to graph libraries used *for solving*: Prim's algorithm for the MST pruning bound is
ten lines in `exact.py` rather than a `networkx` call, for the same reason.

The Tier 1 optimum comes from hand-written combinatorial search — minimal-cover enumeration plus
Held-Karp dynamic programming (see ADR-0009).

## Consequences

- Compliance is structural rather than argued: the dependency list makes the case.
- Tier 1 is bounded by what enumeration can reach (about `|N| ≤ 18`, see ADR-0010) rather than by
  what a solver could reach, so exact results cover a much smaller range of instance sizes.
- Every algorithm in the repository is one this project can be asked to explain.
- The word *solver* is reserved for the thing we do not use, and is never applied to our own code.
