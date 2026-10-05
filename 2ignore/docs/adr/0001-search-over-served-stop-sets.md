# ADR-0001 — The served set is the only decision; an optimum is a minimal cover

Status: accepted

## Context

The model in `MATH.md` §1 has three families of decision variables: `y` (which stops are served),
`x` (which stop each child walks to) and `z` (the arcs the bus traverses). Written that way it
looks like a joint assignment-and-routing search.

Two properties of *this* problem collapse it:

1. **There is no bus capacity constraint.** Once the served set is fixed, every child walks to the
   nearest served stop within `ε`. Any other assignment is either identical in cost or infeasible.
   `x` is therefore derived, not searched.
2. **Distances obey the triangle inequality**, so inserting a stop into a tour can never shorten
   it. A served set containing a redundant stop is therefore never optimal.

## Decision

Every method searches over **subsets of stops**. The assignment is computed by
`solution.assign_children`, and the tour by the routing functions in `tsp.py`. The problem is
treated as a Covering Tour Problem: choose a covering subset, then order it.

"Drop redundant stops" is available to every method as an always-safe improvement move, and the
exact search enumerates minimal covers rather than all `2^|N|` subsets.

## Consequences

- The search space is subsets of `N`, not `(assignment, tour)` pairs — smaller by orders of
  magnitude, and every point in it has a well-defined objective.
- `CoverState`'s removability test becomes the most-used primitive in the project.
- The objective is **not** the number of stops. Classic greedy set cover optimises the wrong
  thing: six clustered stops can beat four scattered ones, so constructions must be route-aware.
- If a future variant adds bus capacity, this ADR is void — `x` becomes a real decision and every
  method needs rewriting, not tuning.
