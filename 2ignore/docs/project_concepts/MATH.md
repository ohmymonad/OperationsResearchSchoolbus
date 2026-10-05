# ScuolaBus I — Mathematical model and lingo

Extracted from `PLAN.md` so a session can load the maths without the rest of the plan.
Symbol → code names live in `CONTEXT.md` (Notation). Method descriptions (H1/H2/H3) stay in `PLAN.md` §2.

## 1. Mathematical model

Sets and data:
- `B` children, residence `p_b ∈ R²`; `N` candidate stops, location `q_n ∈ R²`; school `s`.
- `ε` max walking distance. Cover set `C_b = { n ∈ N : d(p_b, q_n) ≤ ε }`, with `|C_b| ≥ 3` by construction.
- `d(i,j)` Euclidean distance on `N ∪ {s}`.

Variables:
- `y_n ∈ {0,1}` — stop `n` is served by the bus
- `x_{bn} ∈ {0,1}` — child `b` walks to stop `n`
- `z_{ij} ∈ {0,1}` — the bus traverses arc `(i,j)`

Objective: `min Σ_{i,j} d(i,j)·z_{ij}`

Constraints:
1. `Σ_{n ∈ C_b} x_{bn} = 1` ∀`b ∈ B` — every child assigned to exactly one stop
2. `x_{bn} ≤ y_n` ∀`b,n` — only served stops receive children
3. `x_{bn} = 0` if `d(p_b, q_n) > ε` — walking limit
4. degree + connectivity constraints making `z` a single tour over `{s} ∪ { n : y_n = 1 }`

This model is written up for the report, **but it is never handed to a solver** — it exists to
define the objective and to justify the feasibility checks in `solution.py`.

### The structural insight that drives every algorithm

Because there are no bus-capacity constraints, **the assignment `x` is not a real decision**.
Once a set `S ⊆ N` of served stops is chosen and covers every child, each child can simply walk
to their nearest stop in `C_b ∩ S`. So the search space collapses to:

> choose a covering subset `S ⊆ N`, then order it — a **Covering Tour Problem**
> (set covering × TSP, coupled through the objective).

Second consequence, from the triangle inequality: **adding a stop never shortens the tour.**
Therefore an optimal `S` is always a *minimal (irredundant) cover*. This prunes the exact
enumerator massively and gives every heuristic a cheap, always-valid "drop redundant stops"
cleanup move.

A trap worth stating explicitly in the report: minimising the *number* of stops (classic greedy
set cover) is **not** the objective, and correlates with it only loosely. Six clustered stops can
beat four scattered ones. The construction heuristic must be route-aware.

---

## 2. The optimality graph — what is actually possible without a solver

Direct answer to *"when, and if(!), allowed"*: **it is possible, but only honestly in two tiers.**
Without an ILP solver there is no LP relaxation and no Lagrangian bound, so the plan gets ground
truth a different way — by computing it exactly, from scratch.

**Tier 1 — true optimum, small instances (`|N| ≤ 16`, `|B| ≤ 30`).** `exact.py`, hand-written,
no libraries:
- enumerate **minimal covers** by branching on the uncovered child with the fewest options
  (only minimal covers can be optimal — see §1), memoised on `frozenset`
- for each cover, exact TSP by **Held–Karp DP**, `O(2^k·k²)`, capped at `k ≤ 15`
- prune branches whose **MST lower bound** (Prim over `{s} ∪ S_partial`) already exceeds the incumbent

This is exhaustive combinatorial search, not mathematical programming — fully inside the user's
constraint and the course's. It yields a **genuine optimality gap**.

**Tier 2 — medium and large instances.** No optimum is computable. Report `gap = (f − BKS)/BKS`
against the **best-known solution** across all three methods × all seeds, alongside a trivial
valid lower bound `2·max_b min_{n ∈ C_b} d(s,n)` for context. This is standard practice in the
metaheuristic literature and must be **labelled as gap-to-BKS, never as gap-to-optimum** — that
distinction is exactly the kind of thing an OR examiner checks.

The final figure is a two-panel plot: left = gap to true optimum (Tier 1), right = gap to BKS
(Tier 2), three coloured series, error bars over seeds. Note that `exact.py` is **evaluation
infrastructure, not a fourth approach** — say so in the report so it isn't mistaken for one.
