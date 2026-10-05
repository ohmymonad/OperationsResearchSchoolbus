# ADR-0004 — Three methods, from three different families of the syllabus

Status: accepted

## Context

The specification requires at least one resolution method from the course material. Three were
wanted — one per group member — which raises the question of whether three *different* methods
add anything over three tunings of one.

## Decision

H1 **GRASP + VND** (constructive greedy, local search, multi-start), H2 **Tabu Search**
(trajectory metaheuristic with memory), H3 **Memetic Algorithm** (population metaheuristic with
local search). One family each.

Rejected, with reasons:

- **Simulated Annealing** — trajectory method over the same neighbourhood as H2. A second sample
  from one family teaches nothing the first did not.
- **Ant Colony Optimisation** — edge-based by nature, and awkward when the node set itself is what
  varies.
- **Plain VNS** — duplicates the Variable Neighbourhood Descent already inside H1.

## Consequences

- The three genuinely disagree on large instances (~1% spread at 120×45 and above) and agree
  exactly on small ones, which is a result worth reporting rather than a tuning artefact.
- Iteration counts differ by two orders of magnitude at equal wall-clock, which forces ADR-0005.
- Shared machinery has to be neutral between a restart-based, a trajectory-based and a
  population-based search — hence `Budget`, `History` and `Working` living apart from any method.
- H2 and H3 both start from H1's randomised construction. That is deliberate reuse, not a family
  overlap: it guarantees all three are at least as good as the plain greedy solution.
