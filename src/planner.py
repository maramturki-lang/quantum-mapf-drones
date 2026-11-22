"""Planification individuelle : A* sur grille 4-connexe."""
from __future__ import annotations

import heapq
import itertools

from .environment import Grid, Position


def manhattan(a: Position, b: Position) -> int:
    """Heuristique admissible en 4-connexité."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class AStarPlanner:
    """A* classique : plus court chemin d'un agent seul (ignore les autres agents)."""

    def __init__(self, grid: Grid):
        self.grid = grid
        self.nodes_expanded = 0   # mesure de l'effort de recherche

    def heuristic(self, pos: Position, goal: Position) -> int:
        return manhattan(pos, goal)

    def plan(self, start: Position, goal: Position) -> list[Position] | None:
        self.nodes_expanded = 0
        counter = itertools.count()             # départage les égalités de f dans le tas
        open_heap = [(self.heuristic(start, goal), 0, next(counter), start)]
        came_from: dict[Position, Position | None] = {start: None}
        g_score = {start: 0}
        closed: set[Position] = set()

        while open_heap:
            _, g, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            self.nodes_expanded += 1

            if current == goal:
                return self._reconstruct(came_from, current)

            for nxt in self.grid.neighbors(current, include_wait=False):
                new_g = g + 1
                if new_g < g_score.get(nxt, float("inf")):
                    g_score[nxt] = new_g
                    came_from[nxt] = current
                    f = new_g + self.heuristic(nxt, goal)
                    heapq.heappush(open_heap, (f, new_g, next(counter), nxt))
        return None   # cible inaccessible

    @staticmethod
    def _reconstruct(came_from: dict, node: Position) -> list[Position]:
        path = []
        while node is not None:
            path.append(node)
            node = came_from[node]
        return path[::-1]
