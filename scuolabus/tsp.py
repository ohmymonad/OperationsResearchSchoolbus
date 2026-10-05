"""Routing toolkit: constructive heuristics, local search, and exact Held-Karp.

A *tour* here is a list of node ids whose first element is always the school.
The return leg is implicit, so ``[school, a, b]`` denotes the closed cycle
``school -> a -> b -> school``.  Keeping the school pinned at position 0 costs
nothing (a cycle can be rotated freely) and lets the local search skip any move
that would displace the depot.

Nothing in this module knows about children or coverage — it works purely on a
node subset and a distance matrix, so the same code routes candidate solutions
inside every heuristic and inside the exact reference solver.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

#: Improvement threshold; moves gaining less than this are treated as noise.
IMPROVE_TOL = 1e-9

#: Largest number of stops Held-Karp will accept (2**k * k**2 blows up past this).
HELD_KARP_MAX_STOPS = 18


def tour_length(tour: Sequence[int], node_dist: np.ndarray) -> float:
    """Length of the closed tour, including the return leg to ``tour[0]``."""
    if len(tour) < 2:
        return 0.0
    idx = np.asarray(tour, dtype=int)
    return float(node_dist[idx, np.roll(idx, -1)].sum())


# --------------------------------------------------------------- building ---


def nearest_neighbour(stops: Iterable[int], node_dist: np.ndarray, start: int) -> list[int]:
    """Build a tour by repeatedly hopping to the closest unvisited stop."""
    remaining = list(stops)
    if not remaining:
        return [start]
    tour = [start]
    current = start
    while remaining:
        arr = np.asarray(remaining, dtype=int)
        nxt = int(arr[int(np.argmin(node_dist[current, arr]))])
        tour.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return tour


def best_insertion(tour: Sequence[int], node: int, node_dist: np.ndarray) -> tuple[float, int]:
    """Cheapest place to splice ``node`` into ``tour``.

    Returns:
        ``(delta, position)`` where inserting at ``position`` (i.e.
        ``tour[:position] + [node] + tour[position:]``) increases the closed tour
        length by ``delta``.  Position 0 is never returned, so the school stays put.
    """
    idx = np.asarray(tour, dtype=int)
    n = idx.size
    if n == 0:
        return 0.0, 0
    if n == 1:
        return 2.0 * float(node_dist[idx[0], node]), 1
    prev = idx
    nxt = np.roll(idx, -1)
    deltas = node_dist[prev, node] + node_dist[node, nxt] - node_dist[prev, nxt]
    k = int(np.argmin(deltas))
    return float(deltas[k]), k + 1


def cheapest_insertion(stops: Iterable[int], node_dist: np.ndarray, start: int) -> list[int]:
    """Build a tour by always splicing in the stop with the cheapest insertion cost."""
    remaining = set(int(s) for s in stops)
    tour = [start]
    while remaining:
        best = min(
            ((best_insertion(tour, s, node_dist), s) for s in remaining),
            key=lambda item: item[0][0],
        )
        (_, pos), stop = best
        tour.insert(pos, stop)
        remaining.discard(stop)
    return tour


# ----------------------------------------------------------- local search ---


def two_opt(tour: Sequence[int], node_dist: np.ndarray) -> list[int]:
    """Improve a tour by 2-opt segment reversal until no improving move remains.

    Each candidate ``i`` is scanned against all partners ``j > i`` in one
    vectorised numpy expression, which is what keeps this usable inside the
    metaheuristics' inner loops.  The scan is best-improvement per ``i`` and
    sweeps repeatedly until a full pass finds nothing.
    """
    t = np.asarray(tour, dtype=int).copy()
    n = t.size
    if n < 4:
        return t.tolist()

    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            a, b = t[i - 1], t[i]
            js = np.arange(i + 1, n)
            c = t[js]
            d = t[(js + 1) % n]
            deltas = node_dist[a, c] + node_dist[b, d] - node_dist[a, b] - node_dist[c, d]
            k = int(np.argmin(deltas))
            if deltas[k] < -IMPROVE_TOL:
                j = int(js[k])
                t[i : j + 1] = t[i : j + 1][::-1]
                improved = True
    return t.tolist()


def or_opt(tour: Sequence[int], node_dist: np.ndarray, max_segment: int = 3) -> list[int]:
    """Relocate short segments (length 1..``max_segment``), optionally reversed.

    Complements 2-opt: 2-opt cannot move a single stop to a distant part of the
    tour without a costly double reversal, which is exactly the move needed when
    a stop was inserted early by a greedy construction and later turns out to
    belong elsewhere.
    """
    t = list(int(x) for x in tour)
    if len(t) < 4:
        return t

    improved = True
    while improved:
        improved = False
        n = len(t)
        for seg_len in range(1, max_segment + 1):
            if improved:
                break
            for i in range(1, n - seg_len + 1):
                segment = t[i : i + seg_len]
                prev_node = t[i - 1]
                next_node = t[(i + seg_len) % n]
                gain = (
                    node_dist[prev_node, segment[0]]
                    + node_dist[segment[-1], next_node]
                    - node_dist[prev_node, next_node]
                )
                if gain <= IMPROVE_TOL:
                    continue

                rest = t[:i] + t[i + seg_len :]
                m = len(rest)
                best_cost = float("inf")
                best_pos = -1
                best_seg: list[int] = segment
                for pos in range(1, m + 1):
                    left = rest[pos - 1]
                    right = rest[pos % m]
                    for candidate in (segment, segment[::-1]):
                        cost = (
                            node_dist[left, candidate[0]]
                            + node_dist[candidate[-1], right]
                            - node_dist[left, right]
                        )
                        if cost < best_cost:
                            best_cost, best_pos, best_seg = cost, pos, candidate
                if best_cost < gain - IMPROVE_TOL:
                    t = rest[:best_pos] + list(best_seg) + rest[best_pos:]
                    improved = True
                    break
    return t


def local_search(tour: Sequence[int], node_dist: np.ndarray, use_or_opt: bool = True) -> list[int]:
    """Alternate 2-opt and Or-opt until neither improves the tour."""
    t = two_opt(tour, node_dist)
    if not use_or_opt:
        return t
    while True:
        before = tour_length(t, node_dist)
        t = or_opt(t, node_dist)
        t = two_opt(t, node_dist)
        if tour_length(t, node_dist) >= before - IMPROVE_TOL:
            return t


def route(stops: Iterable[int], node_dist: np.ndarray, start: int, polish: bool = False) -> list[int]:
    """Standard evaluation route for a stop set: nearest neighbour then local search.

    ``polish=False`` (2-opt only) is the fast path used to score candidate stop
    sets inside the metaheuristics; ``polish=True`` adds Or-opt and is used when
    a solution is about to be reported.
    """
    tour = nearest_neighbour(stops, node_dist, start)
    return local_search(tour, node_dist, use_or_opt=polish)


# ----------------------------------------------------------------- exact ---


def held_karp(stops: Sequence[int], node_dist: np.ndarray, start: int) -> tuple[float, list[int]]:
    """Exact shortest closed tour through ``stops`` plus ``start``, by DP over subsets.

    Classic Held-Karp: ``dp[mask][i]`` is the cheapest path from ``start`` that
    visits exactly the stops in ``mask`` and ends at stop ``i``.  Runs in
    ``O(2^k k^2)`` time and ``O(2^k k)`` memory, so it is only ever called on the
    small instances used to produce true optimality gaps.

    The transition for a whole ``mask`` is done as one ``(k, k)`` min-plus
    numpy reduction rather than a Python double loop, which is what makes
    ``k = 15`` a second rather than a minute.

    Raises:
        ValueError: if ``len(stops)`` exceeds :data:`HELD_KARP_MAX_STOPS`.
    """
    others = [int(s) for s in stops if int(s) != start]
    k = len(others)
    if k == 0:
        return 0.0, [start]
    if k > HELD_KARP_MAX_STOPS:
        raise ValueError(
            f"Held-Karp called with {k} stops, limit is {HELD_KARP_MAX_STOPS}; "
            "use a heuristic route instead"
        )
    if k == 1:
        return 2.0 * float(node_dist[start, others[0]]), [start, others[0]]

    idx = np.asarray(others, dtype=int)
    sub = node_dist[np.ix_(idx, idx)]
    from_start = node_dist[start, idx]
    to_start = node_dist[idx, start]

    size = 1 << k
    dp = np.full((size, k), np.inf)
    parent = np.full((size, k), -1, dtype=np.int16)
    for i in range(k):
        dp[1 << i, i] = from_start[i]

    for mask in range(1, size):
        row = dp[mask]
        if not np.isfinite(row).any():
            continue
        cand = row[:, None] + sub  # cand[i, j] = cost ending at i, then hop to j
        best_prev = np.argmin(cand, axis=0)
        best_val = cand[best_prev, np.arange(k)]
        for j in range(k):
            if mask >> j & 1:
                continue
            nxt = mask | (1 << j)
            if best_val[j] < dp[nxt, j]:
                dp[nxt, j] = best_val[j]
                parent[nxt, j] = best_prev[j]

    full = size - 1
    closed = dp[full] + to_start
    last = int(np.argmin(closed))
    best_length = float(closed[last])

    order: list[int] = []
    mask, j = full, last
    while j != -1:
        order.append(others[j])
        prev = int(parent[mask, j])
        mask ^= 1 << j
        j = prev
    order.reverse()
    return best_length, [start] + order
