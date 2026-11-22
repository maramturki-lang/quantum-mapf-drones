"""Simulation : exécute un planificateur sur un scénario et mesure ses performances."""
from __future__ import annotations

import copy
import time

from .environment import Agent, Grid, Position
from .metrics import Metrics, compute_metrics


class Simulation:
    def __init__(self, grid: Grid, agents: list[Agent], planner_cls, **planner_kwargs):
        self.grid = grid
        self.agents = copy.deepcopy(agents)     # chaque planificateur part du même scénario
        self.planner = planner_cls(grid, **planner_kwargs)
        self.metrics: Metrics | None = None

    def run(self) -> Metrics:
        t0 = time.perf_counter()
        self.planner.solve(self.agents)
        runtime = time.perf_counter() - t0
        self.metrics = compute_metrics(self.agents, self.planner.name,
                                       runtime, self.planner.nodes_expanded)
        return self.metrics

    @property
    def horizon(self) -> int:
        return max(len(a.path) for a in self.agents)

    def positions_at(self, t: int) -> dict[int, Position]:
        """État du système à l'instant t (utilisé par l'animation)."""
        return {a.agent_id: a.position_at(t) for a in self.agents}
