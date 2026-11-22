"""Environnement MAPF : grille 2D, agents et générateur de scénarios."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

Position = tuple[int, int]  # (ligne, colonne)

# Actions possibles : attendre, haut, bas, gauche, droite (4-connexité)
MOVES = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]


class Grid:
    """Grille 2D avec obstacles statiques."""

    def __init__(self, width: int, height: int, obstacles=None):
        self.width = width
        self.height = height
        self.obstacles: set[Position] = set(obstacles or [])

    def in_bounds(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def is_free(self, pos: Position) -> bool:
        return self.in_bounds(pos) and pos not in self.obstacles

    def neighbors(self, pos: Position, include_wait: bool = True) -> list[Position]:
        """Cases accessibles en un pas de temps (y compris rester sur place)."""
        r, c = pos
        moves = MOVES if include_wait else MOVES[1:]
        return [(r + dr, c + dc) for dr, dc in moves if self.is_free((r + dr, c + dc))]

    def free_cells(self) -> list[Position]:
        return [(r, c) for r in range(self.height) for c in range(self.width)
                if (r, c) not in self.obstacles]

    def is_reachable(self, start: Position, goal: Position) -> bool:
        """BFS : vérifie qu'un chemin existe entre start et goal."""
        if not (self.is_free(start) and self.is_free(goal)):
            return False
        visited, queue = {start}, deque([start])
        while queue:
            cur = queue.popleft()
            if cur == goal:
                return True
            for nxt in self.neighbors(cur, include_wait=False):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    @property
    def obstacle_density(self) -> float:
        return len(self.obstacles) / (self.width * self.height)

    @classmethod
    def random(cls, width: int, height: int, obstacle_ratio: float = 0.15,
               seed: int | None = None) -> "Grid":
        rng = random.Random(seed)
        cells = [(r, c) for r in range(height) for c in range(width)]
        n_obs = int(obstacle_ratio * width * height)
        return cls(width, height, rng.sample(cells, n_obs))

    def __repr__(self) -> str:
        return f"Grid({self.width}x{self.height}, obstacles={len(self.obstacles)})"


@dataclass
class Agent:
    """Un drone : départ, cible et chemin planifié (path[t] = position à l'instant t)."""
    agent_id: int
    start: Position
    goal: Position
    path: list[Position] = field(default_factory=list)

    def position_at(self, t: int) -> Position:
        """Position à l'instant t ; l'agent reste sur sa cible une fois arrivé."""
        if not self.path:
            return self.start
        return self.path[min(t, len(self.path) - 1)]

    @property
    def reached_goal(self) -> bool:
        return bool(self.path) and self.path[-1] == self.goal

    @property
    def travel_distance(self) -> int:
        """Nombre de déplacements réels (les attentes ne comptent pas)."""
        return sum(1 for a, b in zip(self.path, self.path[1:]) if a != b)

    @property
    def arrival_time(self) -> int | None:
        return len(self.path) - 1 if self.reached_goal else None

    @property
    def manhattan_distance(self) -> int:
        """Borne inférieure de la distance (sans obstacles)."""
        return abs(self.start[0] - self.goal[0]) + abs(self.start[1] - self.goal[1])


class ScenarioGenerator:
    """Génère des scénarios valides et reproductibles."""

    def __init__(self, width: int = 10, height: int = 10,
                 obstacle_ratio: float = 0.15, seed: int | None = None):
        self.width = width
        self.height = height
        self.obstacle_ratio = obstacle_ratio
        self.rng = random.Random(seed)

    def generate(self, n_agents: int, max_tries: int = 100) -> tuple[Grid, list[Agent]]:
        for _ in range(max_tries):
            grid = Grid.random(self.width, self.height, self.obstacle_ratio,
                               seed=self.rng.randint(0, 10**9))
            free = grid.free_cells()
            if len(free) < 2 * n_agents:
                continue
            cells = self.rng.sample(free, 2 * n_agents)   # départs et cibles tous distincts
            starts, goals = cells[:n_agents], cells[n_agents:]
            if all(grid.is_reachable(s, g) for s, g in zip(starts, goals)):
                agents = [Agent(i, s, g) for i, (s, g) in enumerate(zip(starts, goals))]
                return grid, agents
        raise RuntimeError("Impossible de générer un scénario valide.")
