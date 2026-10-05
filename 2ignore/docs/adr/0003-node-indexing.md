# ADR-0003 — Stops are nodes `0..|N|-1`, the school is node `|N|`

Status: accepted

## Context

Routing works on a distance matrix over stops *and* the school; covering works on stops alone.
Any indexing scheme has to let both read the same arrays. Putting the school first (node 0, the
usual depot convention in VRP code) means every stop id `n` is node `n+1`, and every conversion
between the two worlds carries an off-by-one.

## Decision

Stops occupy nodes `0 .. |N|-1` and the school is node `|N|`, exposed as `Instance.school_node`
and `Geometry.school_node`. A stop id and its node id are the same integer.

## Consequences

- No conversion exists between stop ids and node ids, so no conversion can be forgotten.
- `node_dist[school, :n_stops]` is exactly "distance from the school to each stop", which the
  lower bound and several heuristics use directly.
- It differs from the depot-first convention a reader may expect from VRP literature, so the
  convention is restated at the top of `instance.py`, `tsp.py` and `CONTEXT.md`.
- Validation checks `0 <= stop < n_stops` before indexing anything, because an out-of-range id is
  now a plausible-looking node id rather than an obvious error.
