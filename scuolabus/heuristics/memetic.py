"""H3 — a Memetic Algorithm: a genetic algorithm with local search inside it.

The chromosome is the served stop set itself, so crossover recombines *covers*
rather than tours.  Almost every recombination of two covers is infeasible —
that is the normal case, not an edge case — so a **repair** operator sits
between crossover and evaluation: greedily add stops until every child is
covered, then drop the ones that recombination made redundant.  Repair is what
keeps a GA competitive on a covering problem; without it, selection spends its
whole budget rediscovering feasibility.

What makes it *memetic* rather than a plain GA is the local-search step: a share
of the offspring are driven to a Variable Neighbourhood Descent local optimum
before they compete, and their improved chromosome is written back (Lamarckian).
The population then searches the space of *local optima*, which is far smaller
than the space of covers.

Fitness comes from :class:`~scuolabus.heuristics.base.TourCache`: many distinct
subsets get evaluated and re-evaluated across generations, and — unlike the
trajectory methods — nothing here needs a tour that evolves monotonically under
node deletion, so memoised routing from scratch is exactly the right tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from scuolabus.cover import CoverState, greedy_cover, greedy_drop_redundant
from scuolabus.distances import Geometry
from scuolabus.heuristics.base import (
    Budget,
    BudgetTracker,
    HeuristicResult,
    History,
    TourCache,
)
from scuolabus.heuristics.grasp import Working, randomized_greedy_construction, vnd
from scuolabus.instance import Instance
from scuolabus.solution import make_solution
from scuolabus.tsp import IMPROVE_TOL

#: Default number of chromosomes kept between generations.
DEFAULT_POPULATION_SIZE = 20

#: Default tournament size — 2 is the classic binary tournament, weak enough
#: pressure that a merely-good individual still reproduces sometimes.
DEFAULT_TOURNAMENT_SIZE = 2

#: Default per-offspring mutation rate.
DEFAULT_MUTATION_RATE = 0.3

#: Default probability that an offspring receives the memetic VND step.
DEFAULT_LOCAL_SEARCH_RATE = 0.35

#: Alphas used to build the initial population.  Higher than GRASP's, because
#: here the point of the construction is spread rather than quality.
INITIAL_ALPHAS = (0.3, 0.5, 0.7, 0.9)

#: Fraction of the served set a full-rate mutation touches.
MUTATION_STRENGTH = 0.25

#: Probability that a mutation flip drops a redundant stop rather than adding one.
DROP_BIAS = 0.5

#: How many of the nearest stops an additive mutation chooses between.
NEIGHBOUR_POOL = 5

#: Multiplier on the number of draws allowed to fill a population or a
#: generation, so the duplicate-rejecting guard can never spin forever.
FILL_ATTEMPTS = 4


@dataclass(frozen=True)
class Individual:
    """One member of the population: a chromosome and its evaluated tour length.

    The tour itself is deliberately absent.  Only the incumbent's tour is ever
    reported, and carrying one per individual would mean copying a route on
    every selection for no benefit.
    """

    served: frozenset[int]
    length: float


# ------------------------------------------------------------- operators ---


def uniform_crossover(
    a: frozenset[int], b: frozenset[int], rng: np.random.Generator
) -> frozenset[int]:
    """Recombine two covers gene by gene, keeping everything the parents agree on.

    Stops served by both parents are inherited unconditionally and stops served
    by neither are never invented, so the child always satisfies
    ``a & b <= child <= a | b``.  Only the *disputed* stops are decided by a coin
    flip.  Agreement is treated as evidence: a stop both parents kept has already
    survived selection twice, and re-rolling it would throw that information
    away — which is the difference between recombination and random restart.
    """
    shared = frozenset(int(s) for s in a & b)
    disputed = sorted(int(s) for s in a ^ b)
    if not disputed:
        return shared

    keep = rng.random(len(disputed)) < 0.5
    return shared | frozenset(stop for stop, take in zip(disputed, keep) if take)


def repair(
    served: Iterable[int], geom: Geometry, rng: np.random.Generator
) -> CoverState:
    """Turn an arbitrary stop set into a feasible, irredundant cover.

    Two passes, in this order:

    1. :func:`~scuolabus.cover.greedy_cover` adds stops until no child is left
       uncovered — crossover and mutation both routinely leave children stranded.
    2. :func:`~scuolabus.cover.greedy_drop_redundant` removes every stop whose
       children are all covered by some other served stop.  Since adding a stop
       can never shorten a tour, a redundant stop is pure cost, so this pass is
       always safe.

    A cover that is already feasible and irredundant is therefore returned
    unchanged, which is what lets an elite chromosome pass through the pipeline
    without being churned.
    """
    state = CoverState(geom, (int(s) for s in served))
    greedy_cover(geom, state, rng)
    greedy_drop_redundant(state, order=sorted(state.served))
    return state


def _nearby_stop(
    geom: Geometry, rng: np.random.Generator, anchor: int, forbidden: set[int]
) -> int | None:
    """Pick a stop near ``anchor``, uniformly among the nearest admissible few."""
    distances = geom.node_dist[anchor, : geom.n_stops]
    pool: list[int] = []
    for stop in np.argsort(distances, kind="stable"):
        stop = int(stop)
        if stop not in forbidden:
            pool.append(stop)
            if len(pool) == NEIGHBOUR_POOL:
                break
    if not pool:
        return None
    return int(rng.choice(pool))


def mutate(
    served: frozenset[int],
    geom: Geometry,
    rng: np.random.Generator,
    rate: float = DEFAULT_MUTATION_RATE,
) -> frozenset[int]:
    """Flip a few genes, biased towards the flips that are worth making.

    A uniform bit flip over ``|N|`` genes is almost always a wasted evaluation:
    dropping an essential stop is undone by repair, and adding a stop on the far
    side of the map is undone by the next drop-redundant pass.  So the two
    directions are biased instead — drops prefer stops that are *already*
    redundant (free to lose), and additions prefer stops geometrically close to
    the ones already served (cheap to insert).

    Each flip touches a stop no earlier flip touched, so a mutation with
    ``rate > 0`` always changes the chromosome instead of silently cancelling
    itself out.

    Raises:
        ValueError: if ``rate`` lies outside ``[0, 1]``.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")

    current = frozenset(int(s) for s in served)
    if rate == 0.0:
        return current

    mutated = set(current)
    state = CoverState(geom, mutated)
    touched: set[int] = set()

    for _ in range(max(1, int(round(rate * MUTATION_STRENGTH * len(current))))):
        redundant = sorted(s for s in mutated - touched if state.is_removable(s))
        if redundant and rng.random() < DROP_BIAS:
            stop = int(rng.choice(redundant))
            mutated.discard(stop)
            state.drop(stop)
        else:
            anchor = int(rng.choice(sorted(mutated))) if mutated else geom.school_node
            stop = _nearby_stop(geom, rng, anchor, mutated | touched)
            if stop is None:
                break
            mutated.add(stop)
            state.add(stop)
        touched.add(stop)

    return frozenset(mutated)


def tournament_select(
    population: list[Individual],
    rng: np.random.Generator,
    size: int = DEFAULT_TOURNAMENT_SIZE,
) -> Individual:
    """Return the shortest of ``size`` individuals drawn without replacement.

    Tournament selection is used rather than fitness-proportionate selection
    because it depends only on the *ranking* of tour lengths.  Lengths here span
    a narrow band around the greedy solution, so proportionate weights would be
    nearly uniform and exert almost no pressure.

    Raises:
        ValueError: if ``population`` is empty.
    """
    if not population:
        raise ValueError("cannot select from an empty population")

    size = max(1, min(int(size), len(population)))
    drawn = rng.choice(len(population), size=size, replace=False)
    return min((population[int(i)] for i in drawn), key=lambda ind: ind.length)


# --------------------------------------------------------------- driver ---


def _rank(individual: Individual) -> tuple[float, list[int]]:
    """Total order on individuals; the chromosome breaks length ties reproducibly."""
    return (individual.length, sorted(individual.served))


def _refine(work: Working, geom: Geometry) -> Working:
    """Drive a working solution to a VND local optimum and polish its tour.

    Every incumbent goes through this, so the recorded best is always a
    *polished* length compared against other polished lengths.  Deferring the
    polish to the end of the run instead would break the guarantee that a longer
    run is never worse than a shorter one: polishing is not order-preserving, so
    a better raw incumbent can polish to a worse tour than a weaker one would.
    """
    refined = work.copy()
    vnd(refined, geom)
    refined.reoptimise(geom, polish=True)
    return refined


def _working(geom: Geometry, cache: TourCache, served: frozenset[int]) -> Working:
    """A working solution for ``served``, reusing the cached tour rather than routing."""
    length, tour = cache.get(served)
    return Working(state=CoverState(geom, served), tour=list(tour), length=length)


def memetic(
    instance: Instance,
    geom: Geometry,
    budget: Budget,
    seed: int = 0,
    population_size: int = DEFAULT_POPULATION_SIZE,
    tournament_size: int = DEFAULT_TOURNAMENT_SIZE,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    local_search_rate: float = DEFAULT_LOCAL_SEARCH_RATE,
    method: str = "memetic",
) -> HeuristicResult:
    """Run the memetic algorithm until ``budget`` is exhausted.

    Generational ``(mu + lambda)`` replacement: parents and offspring compete in
    one pool, duplicate chromosomes are collapsed, and the best ``mu`` survive.
    The deterministic ``alpha = 0`` greedy is planted in the initial population
    and its refinement seeds the incumbent, so — exactly as in H1 and H2 — the
    answer can never be worse than the plain constructive heuristic.

    Raises:
        ValueError: if ``population_size`` is below 2, which leaves nothing to
            recombine and degenerates the search into a random walk.
    """
    if population_size < 2:
        raise ValueError(
            f"population_size must be at least 2 to recombine, got {population_size}"
        )

    rng = np.random.default_rng(seed)
    tracker = BudgetTracker(budget)
    history = History()
    cache = TourCache(geom)

    best = _refine(randomized_greedy_construction(geom, rng, alpha=0.0), geom)
    best_length = best.length

    # An individual's length is the best length yet seen for its chromosome, so
    # the refined incumbent enters at its refined length rather than its
    # cache length.
    population = [Individual(frozenset(best.state.served), best_length)]
    known = {population[0].served}
    for _ in range(FILL_ATTEMPTS * population_size):
        if len(population) >= population_size:
            break
        alpha = float(rng.choice(INITIAL_ALPHAS))
        served = frozenset(randomized_greedy_construction(geom, rng, alpha=alpha).state.served)
        if served in known:
            continue
        known.add(served)
        population.append(Individual(served, cache.length(served)))

    while True:
        # The generation body must never consult the budget: the first k
        # generations of a long run then draw exactly the rng stream of a short
        # run, which is what makes "more generations, never worse" a guarantee.
        chromosomes = {ind.served for ind in population}
        offspring: list[tuple[Individual, Working | None]] = []

        for _ in range(FILL_ATTEMPTS * population_size):
            if len(offspring) >= population_size:
                break
            mother = tournament_select(population, rng, size=tournament_size)
            father = tournament_select(population, rng, size=tournament_size)
            child = uniform_crossover(mother.served, father.served, rng)
            child = mutate(child, geom, rng, rate=mutation_rate)
            served = frozenset(repair(child, geom, rng).served)

            work: Working | None = None
            if rng.random() < local_search_rate:
                work = _working(geom, cache, served)
                vnd(work, geom)
                served = frozenset(work.state.served)  # Lamarckian write-back

            if served in chromosomes:
                continue  # diversity guard: one copy of a chromosome is enough
            chromosomes.add(served)
            length = work.length if work is not None else cache.length(served)
            offspring.append((Individual(served, length), work))

        for individual, work in offspring:
            if individual.length >= best_length - IMPROVE_TOL:
                continue
            candidate = _refine(
                work if work is not None else _working(geom, cache, individual.served), geom
            )
            if candidate.length < best_length - IMPROVE_TOL:
                best, best_length = candidate, candidate.length
                tracker.note_improvement()

        pool = {ind.served: ind for ind in population}
        for individual, _ in offspring:
            previous = pool.get(individual.served)
            if previous is None or individual.length < previous.length:
                pool[individual.served] = individual
        elite = Individual(frozenset(best.state.served), best_length)
        incumbent = pool.get(elite.served)
        if incumbent is None or elite.length < incumbent.length:
            pool[elite.served] = elite
        # The elite has the pool's lowest length, so truncation never drops it.
        population = sorted(pool.values(), key=_rank)[:population_size]

        tracker.tick()
        history.record(tracker.elapsed, tracker.iterations, best_length)
        if tracker.exhausted():
            break

    best.reoptimise(geom, polish=True)
    history.record(tracker.elapsed, tracker.iterations, best.length)

    return HeuristicResult(
        solution=make_solution(geom, best.tour, method=method),
        history=history,
        iterations=tracker.iterations,
        elapsed=tracker.elapsed,
    )
