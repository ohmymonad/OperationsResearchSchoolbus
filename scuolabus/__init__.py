"""ScuolaBus I — school bus stop selection and routing.

Problem 10 of the Ricerca Operativa course project: choose a set of bus stops
covering every child within a walking radius ``epsilon``, and route a single
bus through those stops on a closed tour based at the school, minimising the
tour length.

The package is deliberately free of any LP/ILP solver dependency: all three
resolution methods are heuristics, and the exact reference values used to
measure their quality come from hand-written combinatorial search
(:mod:`scuolabus.exact`).
"""

from scuolabus.instance import Instance, generate_instance, suggest_epsilon

__all__ = ["Instance", "generate_instance", "suggest_epsilon"]
