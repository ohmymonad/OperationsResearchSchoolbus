# Resume State — ScuolaBus I

Handoff for a fresh session. Written at the **start of Phase 5** (the experiment campaign),
immediately before Step 0a of PLAN.md §5a.

---

## 1. Read these first

- **`PLAN.md` in this repo root is the single source of truth.** Any copy under
  `~/.claude/plans/` is stale — ignore it. §0 governs the method, §5a is the exact spec for
  what this session is building.
- `CONTEXT.md` — the project's vocabulary. Use these terms, not synonyms.
- `docs/adr/` — settled decisions, one file each. Do not re-litigate them.

## 2. The one rule that overrides everything

**Test-driven development is mandatory (PLAN.md §0).** For every routine:

1. Write the test first.
2. Run it and watch it fail for the right reason (`ImportError` / `ModuleNotFoundError` counts).
3. Write the minimum implementation.
4. Run again; fix code — or the test if the test was wrong — until green.
5. Only then may the routine be used by other code.

Never weaken a test to reach green. Never leave the suite red while starting new work.
Never claim completion without pasting real pytest output.

## 3. Environment

```bash
cd /home/ernico/codebases/OR
.venv/bin/pytest -o addopts= -q --tb=short     # whole suite; pytest.ini sets testpaths + pythonpath
.venv/bin/pytest tests/test_x.py               # one file
PYTHONPATH=. .venv/bin/python script.py        # ad-hoc scripts need PYTHONPATH
```

Dependencies: numpy, matplotlib, pytest. **No solver, no pandas** — results use stdlib `csv`.

---

## 4. CURRENT STATE

### The suite is green — 559 passed (baseline, before this session's work)

```
.venv/bin/pytest -o addopts= -q --tb=no   ->  559 passed in ~13.6s
```

### Git — clean tree, nothing uncommitted

Latest commit: `e4af492 chore: add local Claude Code skills used by this project's sessions`
(the `skills/` dir, not project content). Before that, `b09cbd0` closed out Phase 5's spec in
PLAN.md. Nothing is in flight.

---

## 5. THE IMMEDIATE NEXT ACTION — Phase 5, Step 0a

Per PLAN.md §5a Step 0: **before a line of `run.py`**, do the two interface changes, in order.

**0a — `Geometry` gains `name`; drop the dead `instance` parameter.**
`grasp`, `tabu_search`, `memetic`, `solve_exact` all take `(instance, geom, …)` today and none
of them read `instance` — everything goes through `Geometry`. `build_geometry` must copy
`instance.name` onto the returned `Geometry`, then all four entry points drop the parameter.
~65 call sites across `scuolabus/` and `tests/` currently pass `instance` positionally or by
keyword — every one needs updating. This is a pure signature change: **the suite must pass
before and after with no test edited except call sites that pass `instance` — no behavioural
test may need editing.** If one does, stop and report; that means the move isn't a pure move.

**0b — move the working-solution substrate out of `grasp.py` into `heuristics/working.py`.**
`Working`, `randomized_greedy_construction`, `drop_redundant_by_saving`, `swap_stop`, `vnd`
currently live in `grasp.py`; `tabu.py` and `memetic.py` import them from there. Pure move, no
signature changes. Two riders, only with 0b:
- give `Working.from_served` an optional `cache` parameter and let `memetic._working` use it
  instead of its private duplicate path.
- narrow `TourCache`'s docstring in `heuristics/base.py` from "all three heuristics" to H3 only
  (H1/H2 edit a carried tour and never re-route a stop set — do NOT wire the cache into them
  to make the old docstring true; see PLAN.md §5a Step 0 and the ADR-0007 rationale).

**Then, in order:** `experiments/families.py` (Step 1) → `experiments/records.py` (Step 2) →
`bounds.BestKnown.from_rows` (Step 3) → `experiments/run.py` (Step 4). Full spec for each is
PLAN.md §5a — do not re-derive it, it's written out in detail there including the exact
`FIELDNAMES` tuple, the family table, and the `--run`/`--grade` split.

### Contracts already fixed by the plan (do not redesign these)

- Tier is *derived* (`n_stops <= HELD_KARP_MAX_STOPS`) and asserted, never a per-row judgement.
- Grading is a **second pass** over the finished CSV, not computed as rows are produced —
  `BestKnown` isn't known until every method × seed has run. Needs `BestKnown.from_rows`.
- `RunRecord.from_heuristic` / `RunRecord.from_exact` are the only two places that build a row;
  a test must assert both emit exactly `FIELDNAMES`.
- Budgets equalise **wall-clock time**, never iteration count.
- Tier 1 gaps use `bounds.optimality_gap` (raises if heuristic beats exact — that's a bug, don't
  catch it). Tier 2 gaps use `bounds.bks_gap`. Never cross the two.

---

## 6. The complete working architecture (as of session start)

| Module | Purpose | Key exports |
|---|---|---|
| `instance.py` | Problem instances | `Instance`, `generate_instance`, `validate_instance` |
| `distances.py` | Distance matrix, coverage | `Geometry`, `build_geometry` |
| `cover.py` | Coverage state | `CoverState`, `greedy_cover`, `greedy_drop_redundant` |
| `tsp.py` | Routing and exact TSP | `held_karp`, `two_opt`, `or_opt`, `route`, `tour_length` |
| `solution.py` | Solution container | `Solution`, `make_solution`, `validate_solution` |
| `heuristics/base.py` | Common infrastructure | `Budget`, `BudgetTracker`, `History`, `TourCache`, `HeuristicResult` |
| `heuristics/grasp.py` | H1: GRASP + VND | `grasp`, `randomized_greedy_construction`, `Working` (moving to `working.py` this session) |
| `heuristics/tabu.py` | H2: Tabu Search | `tabu_search`, `TabuList`, `PenaltyController` |
| `heuristics/memetic.py` | H3: Genetic + VND | `memetic`, `Individual`, `uniform_crossover`, `repair`, `mutate` |
| `exact.py` | Tier 1 ground truth | `solve_exact`, `minimal_covers`, `mst_length`, `EXACT_MAX_STOPS` |
| `bounds.py` | Tier 2 machinery | `trivial_lower_bound`, `Gap`, `optimality_gap`, `bks_gap`, `BestKnown` |
| `experiments/` | **does not exist yet** — this session's work | `families.py`, `records.py`, `run.py` |

**Node convention**: stops are `0..n_stops-1`, school is `n_stops` (`Instance.school_node`).

---

## 7. Keystone invariants (do not break)

- **H3 polishes incumbents at capture.** Other methods polish only at the end. Keep the
  `_refine` call inside H3's loop.
- **H3's generation loop never reads `tracker.elapsed` or `max_iterations`.**
- **`minimal_covers` needs no dedup table.** Disjoint branching makes every cover appear once.
- **`optimality_gap` raises on negative values.** `bks_gap` permits them — asymmetry is
  deliberate.
- **`solve_exact` refuses `max_stops > HELD_KARP_MAX_STOPS`.**
- **Tier 2 gaps always use `bounds.Gap`** and `BestKnown`. Never `optimality_gap` outside Tier 1.
- **`TourCache` is H3-only.** Do not wire it into H1/H2 even to satisfy an old docstring claim.

---

## 8. Commit pattern

Small, progressive, each independently green. Message body is **why**, not what.
Every message ends with: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

Verify each commit standalone:
```bash
git worktree add -q --detach /tmp/verify <sha>
(cd /tmp/verify && .venv/bin/pytest --tb=no -o addopts= -q | tail -1)
git worktree remove --force /tmp/verify
```

---

## 9. Next phases after this one

- **Phase 6** — visualisation (PLAN.md §6): plots for instance maps, solutions, convergence
  curves, box plots, two-panel gap figure, runtime scaling, ε-sensitivity.
- **Phase 7** — report (PLAN.md §7): model, three method descriptions, results, and the
  **mandatory AI-tool disclaimer** (required by course spec).

---

## 10. Session handover protocol

This file is required by **PLAN.md §0** and must be kept current at every phase boundary —
and, for this session specifically, **after every new test is written**, not just at phase
boundaries, so an interruption mid-Phase-5 costs nothing to resume:

- Refresh it before stopping, at interruption, at handover, when a phase completes, and after
  each red→green TDD cycle lands.
- Refresh PLAN.md §8 at the same time for phase-level completions (they are phase-level and
  keystroke-level views of the same state).
- If the suite is red, say why and give the reproducing command.
- Record **why** a design is shaped as it is, especially when shaped by a bug.

Read order for any new session: `MATH.md` (model + lingo), `PLAN.md` (what the project is), then this file (where
it stands and what the next action is).
