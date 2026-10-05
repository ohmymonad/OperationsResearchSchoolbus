# CONTEXT.md — the domain language of ScuolaBus I

This file names the things this project talks about. It is the vocabulary the code, the tests,
the plan and the report all use; when a term here appears in an identifier, a docstring or a
commit message, it means what it says below and nothing else.

Two companion files:

- **`PLAN.md`** — what the project is and how it is built. 
Section 0 (mandatory TDD) governs everything./
- **`docs/adr/`** — decisions already taken and the reasoning behind them, so a later review
  does not re-litigate them.

---

## The problem in one paragraph

Children live at fixed points. A finite set of candidate bus stops is given, and every child has
at least three of them within a walking radius. One bus drives a single closed tour, based at the
school, through some subset of those stops. Choose the subset so that every child has a served
stop within walking distance, and so that the tour is as short as possible. Nothing constrains
bus capacity, so **which** stop a child walks to is not a decision — see *served set* below.

---

## Notation: the model's symbols and their names in code

The mathematical model lives in `MATH.md` §1. This is the map from its symbols to the code.

| Symbol | Term | In code |
|---|---|---|
| `B`, `b` | children, one child | `Instance.children`, child index `b` |
| `p_b` | residence | row `b` of `Instance.children` |
| `N`, `n` | candidate stops, one stop | `Instance.stops`, stop id `n` |
| `q_n` | stop location | row `n` of `Instance.stops` |
| `s` | school | `Instance.school`; as a node, `Instance.school_node` |
| `ε` | walking radius | `Instance.epsilon`, `Geometry.epsilon` |
| `C_b` | option set of child `b` | `Geometry.options[b]` |
| — | cover set of stop `n` | `Geometry.covers[n]` |
| `d(i,j)` | distance between nodes | `Geometry.node_dist[i, j]` |
| — | walking distance | `Geometry.walk[b, n]` |
| `S` | served set | `Solution.served`, `CoverState.served` |
| `y_n` | stop `n` is served | membership in the served set |
| `x_{bn}` | child `b` walks to stop `n` | `Solution.assignment[b] == n` |
| `z_{ij}` | the bus traverses arc `(i,j)` | consecutive entries of `Solution.tour` |

The decision variables `x` and `z` exist in the write-up and nowhere in the code: `x` is derived
(*assignment*) and `z` is a permutation (*tour*).

---

## The instance

**Instance** — one problem to be solved: children, candidate stops, a school, a walking radius,
and a name. Immutable, JSON-serialisable, and always validated on load. `scuolabus/instance.py`.

**Child** — a residence point. Children are never grouped, never weighted, and never optional.

**Candidate stop** — a place the bus *may* stop. Every stop is either **served** or not; there is
no third state. Stops that no solution serves still cost nothing and are simply never visited.

**School** — the start and end of the tour, and also the depot. It is a node like any other in the
distance matrix, and never a stop.

**Walking radius** (`epsilon`, `ε`) — the furthest a child may walk. A hard constraint, not a
term in the objective. Walking distances are *reported* alongside results because a short tour
that serves few stops makes children walk further, but shortening walks is never a goal.

**Option set** — the stops within `ε` of one child (`C_b`). Its size is the child's number of
**options**.

**Three-option requirement** — the course specification demands `|C_b| ≥ 3` for every child
(`MIN_STOPS_PER_CHILD`). Generated instances satisfy it *by construction*; loaded ones are
re-checked by `validate_instance`. See ADR-0006.

**Viable anchor** — a stop with at least two other stops within `2ε`, and therefore a stop the
generator may sample a child around. A necessary but not sufficient condition, used to prune
hopeless anchors cheaply.

**Instance family** — a size class in the experiment campaign (tiny, small, medium, large),
defined by `|B|`, `|N|`, clustering and a tier. A family is a *plan*; an instance is one draw
from it, identified by its seed.

---

## Geometry and coverage

**Geometry** — everything derived from an instance once, up front: the node distance matrix, the
walking-distance matrix, the cover sets and the option sets. Every algorithm reads the geometry
and never re-measures distance itself. `scuolabus/distances.py`.

**Node** — an index into the distance matrix. Stops are nodes `0 .. |N|-1` and the school is node
`|N|`. A stop id and its node id are the same integer, by design (ADR-0003).

**Cover set** — the children within `ε` of one stop: who that stop can serve.

**Cover** — a served set that leaves no child uncovered. "Covering" is the set-covering half of
the problem; "routing" is the TSP half. The two are coupled only through the objective.

**Coverage degree** — how many served stops cover a given child. A child with degree 0 is
**uncovered**; a served set with no uncovered child is **feasible**.

**CoverState** — the incremental bookkeeping of coverage degrees under add and drop. It exists
because recomputing feasibility from scratch after every move would dominate the runtime.
`scuolabus/cover.py`.

**Redundant stop** / **removable stop** — a served stop every one of whose children is also
covered by another served stop. Dropping one is always safe and never lengthens the tour, which
makes it the single most-used improvement move in the project.

**Irredundant** / **minimal cover** — a cover with no redundant stop. Because adding a stop can
never shorten a tour, an optimal served set is always a minimal cover (ADR-0001).

---

## Solutions

**Served set** — the subset of stops the bus visits. **This is the only real decision.** With no
capacity constraint each child simply walks to the nearest served stop in range, so the
assignment is derived, and the whole search space is "which subset, in which order".

**Assignment** — child → served stop, derived by nearest-in-range. Never searched.

**Tour** — a list of node ids beginning with the school, the return leg implicit: `[s, a, b]`
means the closed cycle `s → a → b → s`. The school is pinned at position 0. Never called a
"route order" or a "path"; the routing *functions* build tours.

**Length** — the closed tour length, and the objective value. Minimised. Nothing else is.

**Solution** — a tour, its served set, its assignment, its length, and the name of the method
that produced it. `scuolabus/solution.py`.

**Feasibility oracle** — `validate_solution`, the single source of truth for whether a solution
satisfies the model. Tests, the campaign and the plots all call it, so a bug in a heuristic
surfaces as a validation error rather than a plausible-looking wrong number.

---

## The search

**Method** — one of the three resolution approaches, named in results by the strings `grasp`
(H1), `tabu` (H2) and `memetic` (H3); the exact reference solver reports as `exact`.

**H1 / H2 / H3** — GRASP with Variable Neighbourhood Descent; Tabu Search; a Memetic Algorithm.
Chosen from three different families of the syllabus, deliberately (ADR-0004).

**Working solution** — a served set together with its tour and that tour's length, maintained in
step. Moves edit the carried tour rather than re-routing the set, which is what makes the local
search monotone (ADR-0007).

**Move** — a single edit to the served set: **add**, **drop**, or **swap** (one out, one in).

**Neighbourhood** — the set of solutions one move away. **Local optimum** — a solution no move in
the current neighbourhood improves.

**VND** (Variable Neighbourhood Descent) — cycling the neighbourhoods cheapest-first, restarting
at the cheapest after any success: reoptimise the tour, then drop redundant stops, then swap.

**Polish** — running Or-opt as well as 2-opt on a tour. Reserved for solutions about to be
reported or captured as incumbents; candidate evaluation uses the cheaper 2-opt-only path.

**Incumbent** — the best solution found so far in a run. A run's incumbent can never worsen, and
`History` raises if it does.

**Restart** — one complete construct-then-improve cycle in H1. One H1 iteration is one restart;
one H2 iteration is one move; one H3 iteration is one generation. This is why budgets equalise
time and not iterations (ADR-0005).

**Budget** — the stopping rule of a run: wall-clock seconds, iterations, stagnation, or a
combination. At least one limit is mandatory. **Stagnation** counts iterations since the last
improvement.

**History** — the convergence trace of one run: `(elapsed, iteration, best)` samples, which is
what the convergence plots are drawn from.

**Tour cache** — memoised routing keyed on the served set, so a re-visited subset is not
re-routed. Used by H3, whose population revisits chromosomes constantly.

**Restricted candidate list** and **alpha** (H1) — the randomised greedy admits any candidate
within `alpha` of the best score; `alpha = 0` is a deterministic pure greedy. **Reactive**: alpha
values that have produced good solutions get sampled more often.

**Penalty** (H2) — the adaptive price of an uncovered child, which lets the trajectory cross
infeasible ground. It self-tunes: persistent infeasibility raises it, persistent feasibility
lowers it.

**Tabu tenure**, **aspiration**, **diversification** (H2) — how long a stop stays forbidden after
being moved; the rule that lets a tabu move through when it beats the incumbent; and the
frequency-based term that pushes a stalled search towards rarely-touched stops.

**Chromosome** (H3) — a served set, used as the genetic representation. **Population**,
**generation**, **tournament selection**, **uniform crossover**, **mutation** carry their usual
meanings over that representation.

**Repair** (H3) — turning an arbitrary recombined stop set back into a feasible, irredundant
cover: greedily cover, then drop redundant. Almost every crossover needs it, which is what keeps
a population method competitive on a covering problem.

**Lamarckian write-back** (H3) — when local search improves an offspring, the improved served set
replaces its chromosome, so the population searches the space of local optima.

---

## Evaluation

**Tier 1** — instances small enough for the exact search, where the true optimum is known and a
gap is a genuine **optimality gap**.

**Tier 2** — every instance larger than that. No optimum is computable without a solver, so
quality is measured against the **best known solution** instead.

**Exact** — the enumerate-and-route reference in `scuolabus/exact.py`. It is *evaluation
infrastructure, not a fourth method*: its only job is to make the word "optimal" mean something.

**Minimal-cover enumeration** — the exact search's outer loop: branch on the most constrained
uncovered child, forbidding options already tried, so every minimal cover is produced exactly
once (ADR-0009). Each cover is then routed exactly by **Held-Karp**.

**MST bound** — a minimum spanning tree over the nodes chosen so far, used to prune the
enumeration: a tree is never longer than a tour through a superset of the same nodes.

**Trivial lower bound** — `2 · max_b min_{n ∈ C_b} d(s, n)`: the bus must reach some stop of every
child's option set and come back. Deliberately weak, always valid, and reported as context
beneath Tier 2 results — never as their reference.

**Gap** — a relative distance to a reference, *carrying the kind of reference it used*. There are
exactly two kinds, `"optimum"` and `"best-known"`, and they render their own axis labels. A Tier
2 gap is never called an optimality gap (ADR-0008).

**Best known solution** (BKS) — the shortest tour any method at any seed has found for one
instance. The Tier 2 reference. Held in the `BestKnown` registry, keyed by instance name.

**Campaign** — the whole Phase 5 experiment: families × instances × methods × seeds, at equal
wall-clock budget, producing one flat CSV. **Run** — one (instance, method, seed) triple, one row.

---

## Words that mean something specific here

Use these, and do not substitute:

| Say | Not |
|---|---|
| tour | route order, path, circuit |
| served set | selected stops, chosen stops, solution set |
| cover | covering, hitting set |
| redundant / removable stop | useless stop, extra stop |
| option set | reachable stops, nearby stops |
| cover set | served children, catchment |
| incumbent | current best, champion |
| polish | refine, clean up |
| best-known gap | optimality gap (only Tier 1 is that) |
| method | algorithm, solver, strategy |
| exact | solver, optimiser |

"Route" is a *verb* in this project (`route(stops, ...)` builds a tour) and never a noun.
"Solver" refers to a mathematical-programming library, which this project does not use at all
(ADR-0002) — so it is never used for anything in this repository.
