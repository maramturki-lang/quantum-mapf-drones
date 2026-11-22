"""Extension quantique : sélection de chemins formulée en QUBO, résolue par recuit simulé.

Idée : pour chaque agent i on génère K chemins candidats. La variable binaire
x[i,k] = 1 signifie « l'agent i suit le chemin k ». On cherche l'affectation qui
minimise :

    H(x) = Σ c[i,k] x[i,k]                         (coût : longueur des chemins)
         + A Σ_i (1 - Σ_k x[i,k])²                 (contrainte : un seul chemin par agent)
         + B Σ conflits(i,k ; j,l) x[i,k] x[j,l]   (pénalité : chemins en conflit)

C'est un QUBO : min xᵀ Q x avec x ∈ {0,1}ⁿ. Ce format est exactement celui
accepté par un recuit quantique (D-Wave) ou par QAOA. Ici on le résout avec un
recuit simulé classique (méthode « quantum-inspired »).
"""
from __future__ import annotations

import itertools
import math
import random

import numpy as np

from .coordinator import detect_conflicts
from .environment import Agent, Grid, Position
from .planner import AStarPlanner


# ------------------------------------------------------- Chemins candidats
class CandidatePathGenerator:
    """Génère des chemins variés : chemin A*, chemin qui contourne les cibles des autres agents,
    versions retardées et détours par un point intermédiaire."""

    def __init__(self, grid: Grid, n_candidates: int = 6, max_delay: int = 2,
                 max_stretch: float = 1.5, seed: int = 0):
        self.grid = grid
        self.n_candidates = n_candidates
        self.max_delay = max_delay
        self.max_stretch = max_stretch
        self.astar = AStarPlanner(grid)
        self.rng = random.Random(seed)

    def generate(self, start: Position, goal: Position,
                 avoid: set[Position] | None = None) -> list[list[Position]]:
        base = self.astar.plan(start, goal)
        if base is None:
            return [[start]]
        candidates = [base]
        if avoid:                                               # ne pas traverser les cibles des autres
            blocked = Grid(self.grid.width, self.grid.height, self.grid.obstacles | (avoid - {goal}))
            detour = AStarPlanner(blocked).plan(start, goal)
            if detour and detour != base:
                candidates.append(detour)
        for d in range(1, self.max_delay + 1):                 # attendre d pas au départ
            candidates.append([start] * d + base)
        free = self.grid.free_cells()
        limit = self.max_stretch * len(base) + 2
        for _ in range(50):                                     # détours par un point w
            if len(candidates) >= self.n_candidates:
                break
            w = self.rng.choice(free)
            p1, p2 = self.astar.plan(start, w), self.astar.plan(w, goal)
            if p1 and p2:
                path = p1 + p2[1:]
                if len(path) <= limit and path not in candidates:
                    candidates.append(path)
        return candidates[: self.n_candidates]


# ------------------------------------------------------------ Modèle QUBO
class MAPFQubo:
    """Construit la matrice Q (triangulaire supérieure) du problème de sélection de chemins."""

    def __init__(self, candidates: dict[int, list[list[Position]]],
                 penalty_onehot: float | None = None, penalty_conflict: float | None = None):
        self.candidates = candidates
        self.index = [(i, k) for i, paths in candidates.items() for k in range(len(paths))]
        self.n = len(self.index)
        # Les pénalités doivent dominer tout gain obtenu en violant une contrainte :
        # on les prend supérieures au coût total maximal possible.
        total_max_cost = sum(max(len(p) - 1 for p in paths) for paths in candidates.values())
        self.A = penalty_onehot or total_max_cost + 1
        self.B = penalty_conflict or total_max_cost + 1
        self.Q = np.zeros((self.n, self.n))
        self.offset = 0.0
        self._build()

    @staticmethod
    def _n_conflicts(p: list[Position], q: list[Position]) -> int:
        return len(detect_conflicts([Agent(0, p[0], p[-1], p), Agent(1, q[0], q[-1], q)]))

    def _build(self) -> None:
        Q, A, B = self.Q, self.A, self.B
        # 1) coût de chaque chemin + partie linéaire de la contrainte one-hot
        #    A(1 - Σx)² = A - A Σx + 2A Σ_{k<l} x_k x_l   (car x² = x)
        for v, (i, k) in enumerate(self.index):
            Q[v, v] += (len(self.candidates[i][k]) - 1) - A
        self.offset = A * len(self.candidates)
        for u, v in itertools.combinations(range(self.n), 2):
            (i, k), (j, l) = self.index[u], self.index[v]
            if i == j:                                    # deux chemins du même agent
                Q[u, v] += 2 * A
            else:                                         # chemins de deux agents différents
                c = self._n_conflicts(self.candidates[i][k], self.candidates[j][l])
                if c:
                    Q[u, v] += B * c

    def energy(self, x: np.ndarray) -> float:
        return float(x @ self.Q @ x) + self.offset

    def decode(self, x: np.ndarray) -> tuple[dict[int, list[Position]], bool]:
        """Chemin choisi par agent. valid=False si la contrainte one-hot est violée."""
        chosen, valid = {}, True
        for i, paths in self.candidates.items():
            ks = [k for k in range(len(paths)) if x[self.index.index((i, k))] == 1]
            if len(ks) != 1:
                valid = False
            ks = ks or range(len(paths))                  # réparation : le moins coûteux
            chosen[i] = min((paths[k] for k in ks), key=len)
        return chosen, valid


# --------------------------------------------------------------- Solveurs
class SimulatedAnnealingSolver:
    """Recuit simulé sur un QUBO : retournements de bits acceptés selon Metropolis."""

    def __init__(self, n_sweeps: int = 300, n_reads: int = 10,
                 t_start: float = 10.0, t_end: float = 0.05, seed: int = 0):
        self.n_sweeps, self.n_reads = n_sweeps, n_reads
        self.t_start, self.t_end = t_start, t_end
        self.rng = np.random.default_rng(seed)

    def solve(self, Q: np.ndarray) -> np.ndarray:
        n = Q.shape[0]
        S = Q + Q.T - np.diag(np.diag(Q))                 # forme symétrique pour le calcul de ΔE
        diag = np.diag(Q)
        temps = np.geomspace(self.t_start, self.t_end, self.n_sweeps)
        best_x, best_e = None, math.inf
        for _ in range(self.n_reads):                     # plusieurs départs aléatoires
            x = self.rng.integers(0, 2, n)
            field = S @ x - diag * x                      # Σ_{j≠i} S_ij x_j
            e = x @ Q @ x
            for T in temps:
                for v in self.rng.permutation(n):
                    delta = (1 - 2 * x[v]) * (diag[v] + field[v])
                    if delta <= 0 or self.rng.random() < math.exp(-delta / T):
                        x[v] ^= 1
                        e += delta
                        field += (2 * x[v] - 1) * S[:, v]
                        field[v] -= (2 * x[v] - 1) * S[v, v]
            if e < best_e:
                best_x, best_e = x.copy(), e
        return best_x


class ExhaustiveSolver:
    """Référence exacte : énumère toutes les combinaisons « un chemin par agent »."""

    def solve(self, model: MAPFQubo) -> np.ndarray:
        groups = [[v for v, (i, _) in enumerate(model.index) if i == a] for a in model.candidates]
        best_x, best_e = None, math.inf
        for choice in itertools.product(*groups):
            x = np.zeros(model.n, dtype=int)
            x[list(choice)] = 1
            e = model.energy(x)
            if e < best_e:
                best_x, best_e = x, e
        return best_x


# ------------------------------------------- Planificateur (même interface)
class QUBOPlanner:
    """Interface identique aux autres stratégies : solve(agents)."""
    name = "QUBO + recuit simulé"

    def __init__(self, grid: Grid, n_candidates: int = 6, solver: str = "sa", seed: int = 0):
        self.grid = grid
        self.generator = CandidatePathGenerator(grid, n_candidates, seed=seed)
        self.solver_name = solver
        self.sa = SimulatedAnnealingSolver(seed=seed)
        self.nodes_expanded = 0       # ici : nombre de variables binaires (≈ qubits)
        self.valid = True
        self.model: MAPFQubo | None = None

    def solve(self, agents: list[Agent]) -> list[Agent]:
        goals = {a.goal for a in agents}
        candidates = {a.agent_id: self.generator.generate(a.start, a.goal, avoid=goals)
                      for a in agents}
        self.model = MAPFQubo(candidates)
        self.nodes_expanded = self.model.n
        if self.solver_name == "exact":
            x = ExhaustiveSolver().solve(self.model)
        else:
            x = self.sa.solve(self.model.Q)
        chosen, self.valid = self.model.decode(x)
        for a in agents:
            a.path = chosen[a.agent_id]
        return agents
