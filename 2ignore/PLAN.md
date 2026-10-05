# ScuolaBus I — Three Heuristic Approaches in Python

## Context

`/home/ernico/codebases/OR` started out holding only the problem statement, the teacher's
specification, and a `CLAUDE.md`. Phase 1 of this plan has since been written (see §8).

**`PLAN.md` in this project's root folder is the single source of truth for this project.**
It is the only plan document to read, follow, or update. Any copy living outside the repository
(for example under `~/.claude/plans/`) is a stale artefact and must be ignored — if the two ever
disagree, this file wins. Every change of approach, every deviation, and every status update is
recorded **here**, in the repository, so the plan travels with the code and stays reviewable by
the whole group.

It doubles as the design record for the written report, so the mathematical model in `MATH.md` and the
method descriptions in §2 are meant to be lifted from here into the final write-up.

> **Starting or resuming a session? Read this file, then read `2resume.md` — in that order.**
> `PLAN.md` says what the project is and how it must be built; `2resume.md` says exactly where
> the work stopped, what is committed, what is uncommitted, and what the very next action is.
> Neither replaces the other, and §0's handover protocol keeps `2resume.md` current.

The course project (Problem 10, *ScuolaBus I*) requires: an abstract mathematical model,
**at least one resolution method per group member**, an implementation, and experiments on
at least one instance. This plan builds the whole thing from scratch in Python with three
methodologically distinct heuristics, a results campaign, and a visualisation layer.

Hard constraint set by the user: **all three approaches are heuristics — no LP, no ILP, no
solver library anywhere.** This is stricter than the course rules (which would permit a
solver for error estimation) and the plan respects it.

### Settled decisions
- **Route shape**: closed tour departing from and returning to the school.
- **Metaheuristic family for approach 2**: Tabu Search.
- **Instances**: custom pseudo-random generator (seeded), enforcing the ≥3-stops-within-ε rule.
- **Distances**: Euclidean, but routed through a single distance-matrix module so a road-network
  matrix could be dropped in later without touching the heuristics.

---

## 0. Mandatory working method — test-driven development

**This section governs every other section. It is not advisory.**

No routine enters the program until a test for it exists, has been run, and passes.
The cycle for every single unit of code — helper, method, or module — is:

1. **Write the test first.** It must fail for the right reason before any implementation
   exists (`ImportError` / `AttributeError` counts as failing for the right reason).
2. **Run the test and watch it fail.** `.venv/bin/python -m pytest tests/ -x -q`
3. **Write the minimum implementation** that could make it pass.
4. **Run the tests again.** If they fail, fix the code — or the test, if the test was wrong —
   and re-run. **Do not proceed to the next routine while anything is red.**
5. **Only once green** may the routine be used by other code or wired into the program.

### Non-negotiable rules

- **Never write implementation code before its test.** If an implementation appears first,
  it is deleted or reverted and redone test-first — no exceptions, no "it's obviously correct",
  no "it's just a small helper".
- **Never mark work complete on an unrun test suite.** "It should pass" is not a result;
  paste the actual pytest output.
- **Never weaken a test to make it pass.** If a test is wrong, fix it deliberately and say so.
  Deleting an assertion to get to green is a defect, not a fix.
- **A failing test blocks everything downstream.** No new features while the suite is red.
- **Every bug found later gets a regression test first**, reproducing the bug, before the fix.

### Testing conventions for this project

- `pytest` only; tests live in `tests/`, mirroring the module layout
  (`tests/test_tsp.py` covers `scuolabus/tsp.py`, and so on).
- One behaviour per test; name tests as `test_<unit>_<condition>_<expected>`.
- Shared fixtures (a tiny hand-checkable instance, a built `Geometry`) live in
  `tests/conftest.py` so no test rebuilds them ad hoc.
- **Property-based checks where the property is the real specification**, written as loops over
  seeded random instances: 2-opt never lengthens a tour; incremental `CoverState` always equals a
  from-scratch recomputation; a heuristic never beats the exact optimum.
- **Hand-computed oracles wherever one exists.** Held-Karp on a unit square must return exactly 4;
  a 3-stop instance's optimum is checkable on paper. These catch errors that self-consistency
  checks cannot.
- Randomised tests take an explicit seed and are deterministic; a failure must be reproducible.
- No network, no file-system writes outside `tmp_path`.

### Consequence for the code already written

`scuolabus/{instance,distances,cover,tsp,solution}.py` were written **before** this policy
existed, so they were untested and had to be treated as unverified. Phase 0 retro-fitted the
suite for those five modules and fixed what it exposed. That was a one-off debt from before the
policy existed, not a precedent — everything since has been written test-first.

### Session handover protocol

Work on this project spans sessions, and a session can be interrupted at any moment — including
mid-cycle, with the suite deliberately red at step 2. A fresh session starts with no memory of
any of it, so the state must live in the repository rather than in a conversation.

**`2resume.md` in this project root is the handover file.** It is the second thing to read,
always after this plan. It records, for the exact moment work stopped:

- per-file test counts and which files are green
- **whether the suite is red, and if so whether that red is deliberate** — a TDD step 2 red is
  correct and must not be "fixed" by deleting the failing test file
- the commit list, plus every uncommitted file marked finished or in-progress
- the single next action, with the contracts the already-written tests impose on it
- which files in the directory are *not* project content

**Rules:**

1. **Refresh `2resume.md` before stopping** — when interrupted, when handing off, or whenever
   a phase completes. A stale handover file is worse than none, because it is believed.
2. **Never leave the suite red without saying so in `2resume.md`**, with the reason and the
   command that reproduces it. Red-because-mid-TDD and red-because-broken look identical from
   the outside and must be told apart in writing.
3. **Record the *why*, not just the what.** Design decisions that were reached by discovering a
   bug are the ones a fresh session will otherwise undo — write down what failed and why the
   current shape is the fix.
4. **Update this plan's §8 at the same time.** §8 is the phase-level view; `2resume.md` is the
   keystroke-level view. They are refreshed together or they drift apart.

---

## 1. Mathematical model

Moved to **`MATH.md` §1** (sets, variables, objective, constraints, and the structural insight
that the assignment is not a real decision — the search is over covering stop sets `S`).

---

## 2. The three approaches — and why these three

One per group member, chosen so they belong to three different families in the course syllabus:

| # | Approach | Family | Owns |
|---|----------|--------|------|
| H1 | GRASP + VND | constructive greedy + local search + multi-start | fast good solutions, baseline |
| H2 | Tabu Search | trajectory metaheuristic with memory | intensification/diversification |
| H3 | Memetic (GA + local search) | population metaheuristic | recombination, convergence plots |

Rejected alternatives and why: **Simulated Annealing** overlaps with Tabu (both trajectory,
same neighbourhood — a weak second approach); **ACO** is edge-based by nature and awkward when
the node set itself varies; **plain VNS** duplicates the VND already inside H1.

### Shared machinery (built once, used by all three)

- `tsp.py` — nearest-neighbour and cheapest-insertion construction; **2-opt** and **Or-opt**
  improvement; `tour_length()`. Every heuristic evaluates a candidate `S` by calling
  `route(S, …, polish=False)` (NN + 2-opt); `polish=True` adds Or-opt and is used only when a
  solution is about to be reported.
- `cover.py` — incremental coverage bookkeeping: `cover_degree[b] = |C_b ∩ S|`. A stop is
  *removable* iff every child it covers has `cover_degree ≥ 2`. Add/drop updates are
  `O(children near n)`, never `O(|B|)`. This is the hot path — get it right first.
- **Tour-length cache** keyed by `frozenset(S)`, in `heuristics/base.py`. This was planned as
  machinery for all three methods; it is used by **H3 only**, and that is by design rather than
  by omission. H1 and H2 both carry a `Working` tour and *edit* it (see the Phase 2 note in §8),
  so neither ever re-routes a stop set and neither has anything to look up. H3's population
  decodes chromosomes into stop sets with no tour attached, revisits them constantly, and is the
  one place a memo pays. The module docstring must say H3, not "all three" — an unclaimed
  optimisation reads as an untaken one to the next reader.
- **The working solution** — `Working` (served set + its tour + that tour's length), the
  randomised constructor, and the shared moves — lives in `heuristics/working.py` and is imported
  by all three methods. It is the substrate, not part of H1; see §4.

### H1 — GRASP + VND

*Construction* (randomised, route-aware): start `S = {}`, `tour = [s]`. While children remain
uncovered, score each candidate `n ∉ S` as

```
score(n) = newly_covered(n) / (cheapest_insertion_cost(n, tour) + δ)
```

Build a restricted candidate list from the top-α fraction, pick uniformly at random, insert at
its cheapest position. Then greedily drop redundant stops (largest tour saving first).

*VND*, restarting from neighbourhood 1 on any improvement:
1. 2-opt on the tour
2. Or-opt (relocate segments of length 1–3)
3. drop-redundant
4. swap-stop — replace `n ∈ S` by `n' ∉ S` preserving coverage; gain = removal saving − cheapest insertion cost

*Multi-start*: `R` independent iterations, keep the best. Use **reactive GRASP** (adapt α from
the quality each α has historically produced) — cheap to add, good material for the report.

### H2 — Tabu Search

*State*: the subset `S` (a bitset). *Objective*: `f(S) = tour_len(S) + λ · uncovered(S)`, with
λ adaptive — doubled after `k` consecutive infeasible iterations, halved after `k` feasible ones.
Letting the search cross infeasible ground is what makes this stronger than pure descent.

*Neighbourhood*: add / drop / swap a stop, evaluated over a **sampled candidate list** (size ~30)
rather than the full neighbourhood, so iterations stay cheap.

*Memory*:
- short-term tabu — a dropped stop cannot be re-added for a randomised tenure θ, and vice versa
- aspiration — override tabu if the move beats the global best
- long-term frequency memory — penalise stops used in a large fraction of past iterations when
  the search stagnates; restart from a perturbed copy of the incumbent

*Stopping*: iteration cap, or no improvement for `X` iterations.

### H3 — Memetic Algorithm

*Chromosome*: binary vector over `N`. *Fitness*: `tour_len` after decoding with NN + 2-opt, plus
the same uncovered-children penalty.

- **Init**: population from H1's randomised constructor at high α → diverse, already-feasible covers.
- **Crossover**: uniform, then **repair** — greedily add stops until covered, then drop redundant.
  The repair operator is what keeps the GA competitive here.
- **Mutation**: low-probability bit flips, biased toward dropping redundant stops and adding stops
  geometrically near the current tour.
- **Selection**: binary tournament + elitism.
- **Memetic step**: short VND (2-opt + drop-redundant) on offspring with probability `p_ls`.
- **Diversity guard**: reject duplicate chromosomes by hash.

---

## 3. The optimality graph

Moved to **`MATH.md` §2** (Tier 1 exact ground truth via minimal-cover enumeration + Held–Karp;
Tier 2 gap-to-BKS, always labelled as such, plus the trivial lower bound).

---

## 4. Python architecture

```
scuolabus/
  instance.py        # Instance dataclass, seeded generator, JSON I/O, ≥3-stops validator
  distances.py       # distance matrix, cover sets C_b (pluggable metric)
  solution.py        # Solution dataclass, evaluate(), validate() — the feasibility oracle
  tsp.py             # NN, cheapest insertion, 2-opt, Or-opt, Held-Karp
  cover.py           # incremental coverage state, removable-stop test, redundancy cleanup
  heuristics/
    base.py          # Budget, BudgetTracker, History, TourCache, HeuristicResult
    working.py       # Working, randomised construction, shared moves, VND — the substrate
    grasp.py         # H1
    tabu.py          # H2
    memetic.py       # H3
  exact.py           # minimal-cover enumeration + Held-Karp (Tier 1 ground truth)
  bounds.py          # trivial lower bound, labelled gaps, best-known-solution registry
  experiments/
    families.py      # the declarative family table: sizes, seeds, budget, tier
    records.py       # FIELDNAMES + RunRecord — the results-CSV schema, owned in one place
    run.py           # campaign driver: families × methods × seeds -> results.csv
  viz/               # all plotting (see §6)
tests/
```

### The two seams that Phase 5 must get right

**`heuristics/working.py` is a seam, not a file split.** `Working`, `randomized_greedy_construction`,
`drop_redundant_by_saving`, `swap_stop` and `vnd` were written inside `grasp.py` because H1 needed
them first, and H2 and H3 now import them from there. That makes deleting `grasp.py` break the
other two methods, which is the wrong shape for a project whose central claim is *three distinct
methods*: an examiner reading `import ... from grasp` inside `tabu.py` sees H2 built on H1.
Moving the substrate to `working.py` is a **pure move** — no behaviour changes, no signatures
change — and it makes the dependency graph say what §2 already promises.

**The solver interface is `geom`-first, and `Geometry` carries the case's identity.**
`grasp`, `tabu_search`, `memetic` and `solve_exact` all declare an `instance: Instance` parameter
that **no implementation reads** — every one of them works entirely through `Geometry`. Meanwhile
the campaign needs `instance.name` as the key of the best-known registry, so the driver would
otherwise carry a whole `Instance` alongside a `Geometry` purely to read one string off it, with
nothing checking the two describe the same case. `build_geometry` copies `name` onto `Geometry`,
and the four entry points drop the dead parameter:

```python
grasp(geom, budget, seed=…)          # was grasp(instance, geom, budget, seed=…)
solve_exact(geom, max_stops=…)       # was solve_exact(instance, geom, max_stops=…)
```

Do this **before** `run.py` exists. There are 65 call sites today, all held by the green suite;
the driver, the record module and the figures would each add their own. This is the one change
whose cost strictly grows with every phase it is deferred past.

Dependencies: `numpy`, `matplotlib`, `pytest`. Results tables use the **standard library `csv`
module** rather than `pandas` — the campaign only ever appends flat rows, so pandas earned no
place as a dependency. **No solver, no OR-Tools, no networkx-for-solving** — Prim's algorithm is
ten lines and keeping it in-repo removes any ambiguity about rule 5 of the specification.

The machine has `numpy` and `matplotlib` system-wide but not `pytest`, so the project uses a
`.venv` created with `--system-site-packages`; run everything through `.venv/bin/python`.

All three heuristics share one interface so `run.py` treats them interchangeably and so the
convergence plots can be generated by identical code.

---

## 5. Implementation phases

Every phase below is executed under the §0 cycle: **test first, run it red, implement, run it
green, only then move on.** A phase is not "done" until its tests pass and the output is shown.

**Phase 0 — retro-fit the test suite (blocking).** Write `tests/conftest.py` plus tests for the
five Phase 1 modules, run them, and fix everything they expose. Nothing else starts until this
is green, because every later phase builds directly on these five modules.

**Phase 1 — foundation.** `instance.py`, `distances.py`, `solution.py`, `cover.py`, `tsp.py`.
The generator must place stops and children such that every child provably has ≥3 stops within ε
(generate stops first, then sample each child inside the intersection region of ≥3 stop discs, or
rejection-sample and re-draw). Write `validate()` early — every later phase depends on it.

**Phase 2 — H1 (GRASP + VND).** This also delivers the randomised constructor that H3's
population initialiser and H2's starting point both reuse. Do not start H2 or H3 before the
`tsp.py` local search and the coverage bookkeeping are tested.

**Phase 3 — H2 and H3.** Independent of each other; can be built in parallel by two group members.

**Phase 4 — `exact.py` + `bounds.py`.** Verify it agrees with brute force on a 5-stop instance
before trusting it as ground truth.

**Phase 5 — experiment campaign.** Families × methods × seeds at equal wall-clock, into one flat
CSV. Specified in full below, because Phase 5 writes the first module that consumes all the
others and every interface it fixes is read back by Phase 6.

**Phase 6 — visualisation.** Last, on top of the CSVs.

**Phase 7 — report.** Model, three method descriptions, results, and the mandatory

---

## 5a. Phase 5 in detail — the experiment campaign

Everything here is written test-first under §0. The order matters: the two interface changes come
first, while the suite is the only thing holding the call sites, and the campaign is built on top
of them rather than around them.

### Step 0 — the two interface changes, before a line of `run.py`

| | Change | Shape | Why now |
|---|---|---|---|
| 0a | `Geometry` gains `name`; `grasp`, `tabu_search`, `memetic`, `solve_exact` drop the dead `instance` parameter | mechanical, 65 call sites | The driver needs the instance identity for the BKS key; this hands it over instead of making the campaign carry a second object. Cost strictly grows with every later call site. |
| 0b | `Working`, `randomized_greedy_construction`, `drop_redundant_by_saving`, `swap_stop`, `vnd` move from `grasp.py` to `heuristics/working.py` | pure move, no signature changes | H2 and H3 currently import them *from H1*. Three distinct methods must not look like two built on the first. |

Both are green-to-green: the suite must pass before and after with **no test edited except its
imports**. If a behavioural test needs changing, the move was not a move — stop and report.

Two smaller items ride along with 0b, and only with 0b:

- `Working.from_served` has **no production caller** — `memetic._working` does the same job
  through the `TourCache` instead. Give `from_served` an optional `cache` parameter and let H3
  use it, so one function builds a `Working` from a bare stop set and the cached path is the
  documented one rather than a private helper.
- Narrow the `TourCache` claim in `base.py`'s docstring from "all three heuristics" to H3, per §2.
  **Do not wire the cache into H1 or H2 to make the old claim true.** Neither re-routes a stop
  set — both edit a carried tour, which is the whole point of the Phase 2 decision in §8 — so a
  cache has nothing to offer them, and wiring one in would invalidate every number in §8's table.

### Step 1 — `experiments/families.py`: the tier is a value, not a comment

"Tier 1 uses `optimality_gap`, Tier 2 uses `bks_gap`, never mix them" is this project's sharpest
invariant and currently exists only as prose. One declarative table makes it structural: the
family carries its tier, the tier selects the gap function, and the family name reaches the CSV
so Phase 6 facets without a lookup.

| family | \|B\| × \|N\| | clusters | tier | reference | instance seeds | method seeds |
|---|---|---|---|---|---|---|
| `tiny` | 10 × 8 | — | 1 | exact optimum | 3 | 10 |
| `tiny-wide` | 30 × 18 | — | 1 | exact optimum | 3 | 10 |
| `small` | 30 × 15 | — | 1 | exact optimum | 3 | 10 |
| `medium` | 100 × 40 | 4 | 2 | best known | 3 | 10 |
| `large` | 300 × 100 | 6 | 2 | best known | 3 | 10 |

Each row also carries the remaining `generate_instance` arguments and the wall-clock budget, so
the whole campaign shape is reviewable at a glance and `--quick` is a **filter over rows**
(fewer families, fewer seeds), never a second code path.

Three things changed here against the original family list, all on Phase 4 evidence:

1. **The tier is derived, not declared.** Tier 1 means "an optimum is computable", which is
   exactly `|N| <= HELD_KARP_MAX_STOPS`. Assert it: a row claiming tier 1 at `|N| = 40` must fail
   a test rather than produce a mislabelled axis.
2. **`small` (30 × 15) was always inside the exact ceiling** and the original list assigned it to
   Tier 2 anyway — the list predates `exact.py`. It is Tier 1, and Tier 2 now begins at `medium`.
3. **`tiny-wide` at `|N| = 18` is added** because Phase 4 measured the exact search at 0.096 s for
   `|N| = 16` and 0.178 s for `|N| = 18`. Running it needs `max_stops=HELD_KARP_MAX_STOPS` passed
   deliberately, since `EXACT_MAX_STOPS = 16` is the default guardrail. Cost is driven by the
   number of minimal covers, which grows with `|B|`, so keep `|B|` moderate in the Tier 1 rows.

**Expect Tier 1 to be a table of zeros, and report it as a result.** Phase 4 found all three
methods returning the exact optimum on all ten 20 × 12 instances at a 1 s budget. With `small` and
`tiny-wide` promoted, the honest sentence the campaign can now support is *"all three methods are
provably optimal on every instance up to |N| = 18"* — which is a considerably better finding than
the "we could not separate them" it would otherwise be. The separation lives at `medium` and
`large`, where no optimum exists, and that is precisely why both tiers are needed.

### Step 2 — `experiments/records.py`: one module owns the CSV schema

PLAN §7 requires every figure to regenerate from `results.csv` alone, which makes that file an
interface. Today its shape would be assembled on the way past: `HeuristicResult.summary_row`
contributes five columns plus the three walk statistics, `ExactResult` has no row constructor at
all, `run.py` would add the context columns inline, and `viz/` would read column names from
memory. Four places to keep in step, and drift stays silent until a figure is wrong.

One module owns `FIELDNAMES` and both constructors:

```python
FIELDNAMES = (
    # case identity — from the family table
    "family", "instance", "tier", "n_children", "n_stops", "epsilon",
    "n_clusters", "instance_seed",
    # the run
    "method", "method_seed", "budget_seconds",
    # the result
    "length", "n_served", "iterations", "elapsed",
    "walk_total", "walk_mean", "walk_max",
    # grading — filled by the second pass, empty in a raw row
    "reference", "gap_kind", "gap_percent", "lower_bound",
    # exact runs only, empty for heuristic rows
    "covers_examined", "nodes_pruned",
)

RunRecord.from_heuristic(result, geom, case) -> RunRecord
RunRecord.from_exact(result, geom, case)     -> RunRecord
```

`gap_kind` is `bounds.Gap.kind` — `"optimum"` or `"best-known"` — carried into the file so Phase 6
reads its axis label off the data instead of deciding it. That is ADR-0008 surviving the trip to
the plot. A test asserts every `RunRecord` writes exactly `FIELDNAMES`, so schema drift becomes a
test failure rather than a wrong axis.

Exact runs are rows in the same file, not a sidecar. They are the Tier 1 reference, and
`covers_examined` / `nodes_pruned` are report material for "why does Tier 1 stop where it stops".

### Step 3 — grading is a second pass, and the campaign is resumable

The Tier 2 reference is only known once every method × seed has run, so gaps cannot be computed
as rows are produced. **Settled: the driver appends raw rows as it goes, and a separate grading
pass fills the gap columns from the finished file.**

```
run.py  --run    : for each (case, method, seed) -> append one raw row      (gap columns empty)
run.py  --grade  : read results.csv -> BestKnown.from_rows -> fill gaps -> atomic replace
run.py           : both, in order
```

This needs one addition to `bounds.py`: **`BestKnown.from_rows`, the mirror of the existing
`rows()`**. Today the registry writes out and nothing reads back, so the Tier 2 reference exists
only inside a process that must survive the entire campaign — a crash at run 24 of 30 loses it and
the campaign restarts from nothing. With a load seam, the registry has two adapters (in-memory for
tests, CSV for the campaign), gaps become a **pure function of the CSV**, Phase 6 can re-grade
without re-running anything, and `--resume` is just "skip the `(instance, method, seed)` triples
already in the file".

Rules for the grading pass:

- Tier 1 rows take their reference from the `exact` row for the same instance, via
  `bounds.optimality_gap`. It raises on a heuristic below the optimum — that is the keystone
  cross-check firing, and the pass must not catch it.
- Tier 2 rows take their reference from `BestKnown.from_rows`, via `bounds.bks_gap`. Never
  `optimality_gap`, and the tier comes from the family table, not from a judgement at the call site.
- `lower_bound` is `bounds.trivial_lower_bound`, reported on every row as context. Phase 4 measured
  it at ~63% of the true optimum, so it is a floor under the plot and never a gap reference.
- Write via a temporary file and an atomic replace, so an interrupted grade cannot truncate a
  campaign that took half an hour to produce.

### Step 4 — the driver itself

`python -m scuolabus.experiments.run --quick` must produce `results.csv`, per §7. Equalise
**wall-clock time, not iterations**: at a 2 s budget H2 takes ≈1200 cheap moves, H1 ≈28 full
restarts and H3 ≈7 generations of 20, so an equal-iteration comparison would compare nothing. The
budget lives in the family table, one value per family, identical across the three methods.

Each run also writes its `History` to a per-run convergence file — that trace is the only source
for §6's convergence curves, and it cannot be recovered from `results.csv`.

`results/` and `*.csv` are already git-ignored; the campaign output is reproducible from a seed,
so it is regenerated rather than committed.

---

## 6. Visualisation deliverables

1. **Instance map** — children as dots, candidate stops as hollow squares, school as a star,
   ε-discs shaded for a sample of children.
2. **Solution map** — served stops filled, tour as a directed polyline, thin grey spokes from each
   child to its assigned stop. One panel per method, shared axes, for visual comparison.
3. **Convergence curves** — objective vs. elapsed time (not iterations — the three methods have
   wildly different iteration costs), median line with an inter-quartile band across seeds.
4. **Box plots** of final tour length per method per instance family.
5. **The optimality/gap figure** — the two-panel plot described in `MATH.md` §2.
6. **Runtime vs. instance size**, log-log.
7. **ε-sensitivity curve** — tour length vs. ε. Expect a monotone decrease with diminishing
   returns; it is the most interesting *operations-research* result in the project and worth a
   paragraph of its own in the report.

---

## 7. Verification

The working method is defined in **§0 and is mandatory**; this section lists what the suite must
end up covering, not permission to write it afterwards.

- `.venv/bin/python -m pytest tests/ -q` — unit tests on: `validate()` rejecting an over-ε assignment, an uncovered
  child, and a tour that skips a served stop; 2-opt never increasing length; incremental
  coverage state matching a recomputed-from-scratch state after random add/drop sequences.
- **Hand-checkable instance**: 6 children, 5 stops, drawn on paper; assert all three heuristics
  and `exact.py` agree on the optimum.
- **Cross-validation**: on ~20 random tiny instances, assert `heuristic ≥ exact` always (a
  heuristic beating the "exact" optimum means the enumerator is buggy — this is the single most
  valuable test in the suite).
- **Determinism**: same seed → identical solution, for all three methods.
- End-to-end: `python -m scuolabus.experiments.run --quick` produces `results.csv`, then the viz
  scripts regenerate every figure from that CSV alone.

Phase 5 adds four gates of its own, each guarding one of the interfaces §5a fixes:

- **The move is a move.** After step 0 the suite passes with no test edited except its imports.
  A behavioural test that needs changing means the refactor changed behaviour — stop and report.
- **The tier is not a judgement.** Every family row claiming tier 1 satisfies
  `n_stops <= HELD_KARP_MAX_STOPS`; a row that does not must fail this test rather than reach a
  plot. And `bounds.optimality_gap` is never reachable from a tier-2 row — assert the dispatch,
  not the intention.
- **The schema is single-sourced.** Every `RunRecord` writes exactly `FIELDNAMES`, from both
  constructors, so a column added in one place and forgotten in the other is a test failure.
- **Grading is a pure function of the file.** Grading a results file twice is a no-op, and
  `BestKnown.from_rows(registry.rows())` round-trips to an equal registry. This is what lets a
  campaign resume after a crash and lets Phase 6 re-grade without re-running the search.

---

## 8. Implementation status

| Phase | State |
|---|---|
| 0 — retro-fit test suite | done |
| 1 — foundation | verified by Phase 0 |
| 2 — H1 GRASP + VND | **done** — 50 tests |
| 3 — H2 Tabu, H3 Memetic | **done** — H2 (32 tests), H3 (33 tests) |
| 4 — `exact.py`, `bounds.py` | **done** — 76 + 48 tests |
| 5 — experiment campaign | not started — specified in §5a |
| 6 — visualisation | not started |
| 7 — report | not started |

**559 tests green, whole suite.** Phases 3 and 4 are closed: all three resolution methods
exist, the exact solver that grades them exists, and everything is committed. `2resume.md` holds
the keystroke-level state and the next action, which is Phase 5 (the experiment campaign).

The domain vocabulary now lives in `CONTEXT.md` and the settled design decisions in `docs/adr/`,
one file each. §5a was written against the architecture review of 2026-08-26 (`review reports/`),
whose six candidates were re-checked against the code before being folded in; the review's own
`TourCache` candidate is answered in §5a step 0 by narrowing the claim rather than wiring the
cache in, and its "small is Tier 2" assumption was corrected — see deviation 7.

Measured at a 2-second budget per method, against the deterministic greedy construction:

| Instance | greedy | H1 GRASP | H2 Tabu | H3 Memetic |
|---|---|---|---|---|
| 30×15 | 127.4 | 127.2 | 127.2 | 127.2 |
| 60×25 | 214.5 | 199.2 | 199.2 | 199.2 |
| 120×45, 4 clusters | 352.4 | **271.0** | 278.6 | **271.0** |
| 300×100, 6 clusters | 456.7 | **318.1** | 319.2 | 320.2 |

The three agree on the small instances and separate by ~1% on the large ones, which is the
expected shape: they are methodologically distinct, not differently-tuned versions of one search.
Iteration counts differ by two orders of magnitude at equal wall-clock (H2 ≈ 1200 cheap moves,
H1 ≈ 28 full restarts, H3 ≈ 7 generations of 20), which is exactly why §5 equalises *time*.

Run the suite with `.venv/bin/pytest` from the repository root (`pytest.ini` sets `testpaths`
and `pythonpath`). Ad-hoc scripts still need `PYTHONPATH=.`.

Test files, mirroring the modules they cover: `tests/conftest.py` (fixtures plus the
brute-force permutation oracle), `test_tsp.py` (128), `test_grasp.py` (50), `test_cover.py` (44),
`test_instance.py` (43), `test_base.py` (42), `test_solution.py` (37), `test_memetic.py` (33),
`test_tabu.py` (32), `test_distances.py` (26) — plus `test_exact.py` (76) and `test_bounds.py`
(48) from Phase 4.

**What Phase 0 uncovered.** Two defects, both fixed:

1. *A real bug in `validate_solution`* — it recorded an out-of-range stop id as a problem and
   then kept going, crashing with `IndexError` inside `tour_length` instead of raising
   `SolutionError`. Every check after the id test indexes the distance matrices by those ids, so
   the validator now raises immediately once any id is out of bounds, and it additionally
   validates ids appearing in `assignment` (which had no check at all). This is exactly the
   failure mode the oracle exists to catch, and it was only found by feeding it invalid input.
2. *A wrong test* — a "3-4-5 triangle" oracle in `test_distances.py` whose legs were actually
   5 and 4, so the expected value should have been √41. Corrected in the test, not worked around
   in the code.

**What Phase 2 changed by discovery.** The VND neighbourhoods were first designed to score
candidates by re-routing a stop set through `TourCache`. That design cannot honour the
"never lengthens the tour" contract: dropping a stop is only guaranteed non-increasing for an
*optimal* tour, and NN+2-opt is not monotone under node deletion. The neighbourhoods now carry
and edit a `Working` tour, which is monotone by the triangle inequality. `TourCache` remains
correct for population fitness in H3, where monotonicity is not required. This is the decision
most likely to be silently reverted by a session that does not know why it exists.

**What Phase 3 changed by discovery.** Two things, both worth keeping:

1. *H3 polishes each incumbent when it is captured, not once at the end.* H1 and H2 defer the
   Or-opt polish to the final line, which is fine for them but would break H3's guarantee that
   more generations never produce a worse answer: polishing is not order-preserving, so a better
   raw incumbent can polish to a longer tour than a weaker one would. Comparing polished against
   polished restores the guarantee.
2. *A written test asserted a false premise.* `test_repair_preserves_an_already_minimal_cover`
   built its "minimal" cover from the α=0 greedy — but that construction is **not** minimal: a
   stop added early is often made redundant by one added later, and on that fixture five of its
   nine stops were removable. As written it contradicted the neighbouring test requiring `repair`
   to always return an irredundant cover, so no correct implementation could pass both. The setup
   now prunes the cover first and asserts that it did. Per §0 this is recorded rather than quietly
   patched — and it is the same class of defect as the √41 triangle Phase 0 found, which is why
   §0 requires hand-checked oracles to be checked by hand.

**What Phase 4 measured.** `exact.py` now grades the heuristics, and the first Tier 1 numbers are
worth stating plainly.

*The heuristics are already optimal at this scale.* On ten seeded 20×12 instances at a 1-second
budget, all three methods returned **the exact optimum on every single one** — mean optimality gap
0.00%, max 0.00%. That is a real result, not a broken test: the keystone cross-check is not
vacuous, because on twenty 10×8 instances at a deliberately starved 5-iteration budget the plain
greedy construction sits strictly above the optimum on five and Tabu on three. The inequality
constrains something; the methods simply clear it. **Tier 1 therefore cannot separate the three
methods** — the separation in §8's 2-second table only appears at 120×45 and above, where no
optimum is computable. The report should say this rather than present ten identical zeros as
though they were a comparison.

*The trivial lower bound is loose.* It sits at ~63% of the true optimum across those instances.
That is expected — it prices one child's round trip and ignores everyone else — and it is why `MATH.md` §2
uses it as context beneath the Tier 2 gaps rather than as the reference they are measured against.

*The exact search is far cheaper than assumed.* Cost is dominated by the number of minimal covers,
not by `2^|N|`:

| \|N\| | minimal covers | nodes pruned | seconds |
|---|---|---|---|
| 8 | 16 | 3 | 0.004 |
| 12 | 40 | 9 | 0.013 |
| 14 | 70 | 49 | 0.034 |
| 16 | 202 | 34 | 0.096 |
| 18 | 312 | 30 | 0.178 |

So `EXACT_MAX_STOPS = 16` is a conservative guardrail, not a performance wall — the real ceiling
is `HELD_KARP_MAX_STOPS = 18`, because a cover could in the worst case contain every stop. Phase 5
may raise the Tier 1 family to `|N| = 18` if a wider exact family is wanted; going beyond that
needs a per-cover size check, which nothing currently requires. Note also that *loosely* constrained
instances are the expensive ones — at `|N| = 16`, `|B| = 10` yields 37 covers but `|B| = 60` yields
216 — the opposite of the intuition that more children means more work.

### Deviations from the plan as originally approved

Recorded here so the report matches the code:

0. **The Phase 1 modules were written before their tests**, which violates §0. They are treated
   as unverified until Phase 0 retro-fits the suite and turns it green. This is a one-off debt
   from before the policy existed, not a precedent — everything after it is written test-first.
1. **`pandas` dropped** in favour of the stdlib `csv` module — see §4.
2. **H3 evaluates only feasible chromosomes.** §2 specifies a memetic fitness of
   `tour_len + uncovered-children penalty`. The implementation repairs every chromosome to
   feasibility before evaluating it, so the penalty term is never reachable and was dropped
   rather than left as dead code. Infeasible search is what H2 does — leaving it out of H3 keeps
   the two methods genuinely distinct, which is the point of running three.
3. **The §7 hand-checkable instance is 6 stops, not 5.** The spec requires three stops within ε
   of every child, so two separated clusters need three stops each; five cannot be arranged into
   two clusters without collapsing them into one, which would make the routing trivial. The
   fixture in `tests/conftest.py` uses six, with optimum `12 + 6√2` verified independently by
   exhaustive search.
4. **2-opt uses vectorised numpy sweeps, not don't-look bits.** Don't-look bits are conventionally
   held per *city*, but every 2-opt move reverses a segment and so renumbers positions; keeping the
   bits correct under reversal needs a position-lookup table and is a classic source of subtle bugs.
   Scanning all partners `j` for a given `i` in one numpy expression delivers most of the same
   speed-up at the tour sizes here (≤ ~150 stops) with far less that can go wrong. Revisit only if
   profiling shows 2-opt dominating the metaheuristics' runtime.
5. **The solver entry points are `geom`-only.** `grasp`, `tabu_search`, `memetic` and
   `solve_exact` were written taking `(instance, geom, …)`, and no implementation ever read
   `instance` — every one of them works entirely through `Geometry`. `Geometry` now carries the
   instance `name`, so it is the complete case handle the campaign keys its best-known registry
   on, and the dead parameter is gone. Recorded because §4's original interface sketch says
   `solve(instance, seed, budget)`.

6. **`heuristics/working.py` did not exist in the original tree.** The shared working solution and
   its moves were written inside `grasp.py` because H1 needed them first; H2 and H3 then imported
   them from H1's module, which made the dependency graph misrepresent the project's central claim
   of three distinct methods. §2 always promised this as shared machinery — the file makes the
   seam match the promise. Pure move, no behaviour change.

7. **`small` (30 × 15) is a Tier 1 family, not Tier 2.** The original family list in §5 was written
   before `exact.py` existed and assigned every family above `tiny` to Tier 2 by size intuition.
   `|N| = 15` is inside the exact ceiling, so that instance has a computable optimum and reporting
   a best-known gap for it would have understated what the project can prove. Tier is now derived
   from `|N| <= HELD_KARP_MAX_STOPS` and asserted, and Tier 2 begins at `medium`.

8. **Gaps are computed in a second pass, not as rows are produced.** The Tier 2 reference is not
   known until every method × seed has run. The driver therefore appends raw rows and a grading
   pass fills the gap columns from the finished file, which additionally makes the campaign
   resumable and makes Phase 6 able to re-grade without re-running any search. This requires
   `BestKnown.from_rows` as the mirror of the existing `rows()`; `MATH.md` §2 described the registry as
   in-process only.
