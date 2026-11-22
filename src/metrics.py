"""Métriques de performance d'une solution MAPF."""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .coordinator import detect_conflicts
from .environment import Agent


@dataclass
class Metrics:
    planner: str
    n_agents: int
    total_distance: int        # somme des déplacements réels (sans les attentes)
    n_conflicts: int           # conflits de sommet + d'arête
    success_rate: float        # % d'agents arrivés à destination
    makespan: int              # temps de mission : instant où le dernier agent termine
    sum_of_costs: int          # somme des temps d'arrivée (attentes comprises)
    runtime_ms: float          # temps d'exécution de l'algorithme
    nodes_expanded: int        # effort de recherche

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(agents: list[Agent], planner_name: str,
                    runtime_s: float, nodes_expanded: int) -> Metrics:
    return Metrics(
        planner=planner_name,
        n_agents=len(agents),
        total_distance=sum(a.travel_distance for a in agents),
        n_conflicts=len(detect_conflicts(agents)),
        success_rate=100 * sum(a.reached_goal for a in agents) / len(agents),
        makespan=max(len(a.path) - 1 for a in agents),
        sum_of_costs=sum(len(a.path) - 1 for a in agents),
        runtime_ms=1000 * runtime_s,
        nodes_expanded=nodes_expanded,
    )
