"""Coordination multi-agents : détection de conflits et planification prioritaire."""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

from .environment import Agent, Grid, Position
from .planner import AStarPlanner


# ---------------------------------------------------------------- Conflits
@dataclass(frozen=True)
class Conflict:
    kind: str                  # "vertex" (même case) ou "edge" (échange de cases)
    agents: tuple[int, int]
    time: int
    location: tuple


def detect_conflicts(agents: list[Agent]) -> list[Conflict]:
    """Compare les chemins deux à deux, à chaque instant t."""
    conflicts = []
    horizon = max((len(a.path) for a in agents), default=0)
    for t in range(horizon):
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                pa, pb = a.position_at(t), b.position_at(t)
                if pa == pb:
                    conflicts.append(Conflict("vertex", (a.agent_id, b.agent_id), t, pa))
                elif t + 1 < horizon:
                    na, nb = a.position_at(t + 1), b.position_at(t + 1)
                    if pa == nb and pb == na:
                        conflicts.append(Conflict("edge", (a.agent_id, b.agent_id), t, (pa, pb)))
    return conflicts


# ------------------------------------------------------ Table de réservation
class ReservationTable:
    """Mémorise les cases (x, y, t) et les arêtes déjà occupées par les agents planifiés."""

    def __init__(self):
        self.vertices: set[tuple[Position, int]] = set()
        self.edges: set[tuple[Position, Position, int]] = set()
        self.parked: dict[Position, int] = {}   # cible -> instant à partir duquel l'agent y reste

    def reserve_path(self, path: list[Position]) -> None:
        for t, pos in enumerate(path):
            self.vertices.add((pos, t))
            if t + 1 < len(path):
                self.edges.add((pos, path[t + 1], t))
        self.parked[path[-1]] = len(path) - 1

    def is_free(self, cur: Position, nxt: Position, t: int) -> bool:
        """Le mouvement cur (instant t) -> nxt (instant t+1) est-il autorisé ?"""
        if (nxt, t + 1) in self.vertices:                       # conflit de sommet
            return False
        if nxt in self.parked and t + 1 >= self.parked[nxt]:    # agent garé sur sa cible
            return False
        if (nxt, cur, t) in self.edges:                         # conflit d'arête (échange)
            return False
        return True

    def last_time_used(self, pos: Position) -> int:
        return max((t for p, t in self.vertices if p == pos), default=-1)


# ------------------------------------------------------- A* espace-temps
class SpaceTimeAStar(AStarPlanner):
    """A* dans l'espace (position, temps) : peut attendre et évite les réservations."""

    def plan(self, start, goal, reservations: ReservationTable | None = None,
             max_time: int | None = None):
        reservations = reservations or ReservationTable()
        max_time = max_time or 4 * self.grid.width * self.grid.height
        earliest_goal = reservations.last_time_used(goal) + 1   # la cible doit rester libre ensuite
        self.nodes_expanded = 0
        counter = itertools.count()
        start_state = (start, 0)
        open_heap = [(self.heuristic(start, goal), next(counter), start_state)]
        came_from = {start_state: None}
        closed = set()

        while open_heap:
            _, _, state = heapq.heappop(open_heap)
            if state in closed:
                continue
            closed.add(state)
            self.nodes_expanded += 1
            pos, t = state

            if pos == goal and t >= earliest_goal:
                return [p for p, _ in self._reconstruct(came_from, state)]
            if t >= max_time:
                continue

            for nxt in self.grid.neighbors(pos, include_wait=True):   # l'attente est permise
                if not reservations.is_free(pos, nxt, t):
                    continue
                new_state = (nxt, t + 1)
                if new_state not in came_from:      # g = t : la première découverte est optimale
                    came_from[new_state] = state
                    f = (t + 1) + self.heuristic(nxt, goal)
                    heapq.heappush(open_heap, (f, next(counter), new_state))
        return None


# ------------------------------------------------------------- Stratégies
class IndependentPlanner:
    """Baseline : chaque agent fait son A* sans tenir compte des autres."""
    name = "A* indépendant"

    def __init__(self, grid: Grid):
        self.planner = AStarPlanner(grid)
        self.nodes_expanded = 0

    def solve(self, agents: list[Agent]) -> list[Agent]:
        self.nodes_expanded = 0
        for a in agents:
            a.path = self.planner.plan(a.start, a.goal) or [a.start]
            self.nodes_expanded += self.planner.nodes_expanded
        return agents


class PrioritizedPlanner:
    """Planification prioritaire : agents planifiés un par un avec table de réservation."""
    name = "A* espace-temps prioritaire"

    def __init__(self, grid: Grid, priority: str = "longest_first"):
        self.planner = SpaceTimeAStar(grid)
        self.priority = priority
        self.nodes_expanded = 0

    def order(self, agents: list[Agent]) -> list[Agent]:
        if self.priority == "longest_first":
            return sorted(agents, key=lambda a: -a.manhattan_distance)
        if self.priority == "shortest_first":
            return sorted(agents, key=lambda a: a.manhattan_distance)
        return list(agents)                         # ordre des identifiants

    def solve(self, agents: list[Agent]) -> list[Agent]:
        table = ReservationTable()
        self.nodes_expanded = 0
        for a in self.order(agents):
            path = self.planner.plan(a.start, a.goal, table)
            self.nodes_expanded += self.planner.nodes_expanded
            a.path = path if path is not None else [a.start]   # échec : l'agent reste au départ
            table.reserve_path(a.path)
        return agents
