"""Tests unitaires : environnement, A*, détection de conflits, coordination, métriques."""
import pytest

from src.environment import Agent, Grid, ScenarioGenerator
from src.planner import AStarPlanner, manhattan
from src.coordinator import (IndependentPlanner, PrioritizedPlanner, ReservationTable,
                             SpaceTimeAStar, detect_conflicts)
from src.simulation import Simulation


# ------------------------------------------------------------ Environnement
def test_neighbors_respect_bounds_and_obstacles():
    grid = Grid(3, 3, obstacles=[(0, 1)])
    assert set(grid.neighbors((0, 0), include_wait=False)) == {(1, 0)}
    assert (0, 0) in grid.neighbors((0, 0), include_wait=True)      # attendre est permis


def test_reachability():
    wall = Grid(3, 3, obstacles=[(0, 1), (1, 1), (2, 1)])           # mur vertical
    assert not wall.is_reachable((0, 0), (0, 2))
    assert Grid(3, 3).is_reachable((0, 0), (2, 2))


def test_scenario_generator_is_valid_and_reproducible():
    g1, a1 = ScenarioGenerator(10, 10, 0.15, seed=7).generate(5)
    g2, a2 = ScenarioGenerator(10, 10, 0.15, seed=7).generate(5)
    assert g1.obstacles == g2.obstacles                               # même graine -> même scénario
    assert [(a.start, a.goal) for a in a1] == [(a.start, a.goal) for a in a2]
    cells = [a.start for a in a1] + [a.goal for a in a1]
    assert len(set(cells)) == 10                                      # départs et cibles distincts
    assert all(g1.is_free(c) for c in cells)
    assert all(g1.is_reachable(a.start, a.goal) for a in a1)


# ----------------------------------------------------------------------- A*
def test_astar_finds_shortest_path_without_obstacles():
    path = AStarPlanner(Grid(5, 5)).plan((0, 0), (4, 4))
    assert path[0] == (0, 0) and path[-1] == (4, 4)
    assert len(path) - 1 == manhattan((0, 0), (4, 4))


def test_astar_goes_around_obstacles():
    grid = Grid(3, 3, obstacles=[(0, 1), (1, 1)])
    path = AStarPlanner(grid).plan((0, 0), (0, 2))
    assert len(path) - 1 == 6                                          # détour par la ligne 2
    assert all(grid.is_free(p) for p in path)
    assert all(manhattan(a, b) == 1 for a, b in zip(path, path[1:]))   # pas unitaires


def test_astar_returns_none_when_unreachable():
    wall = Grid(3, 3, obstacles=[(0, 1), (1, 1), (2, 1)])
    assert AStarPlanner(wall).plan((0, 0), (0, 2)) is None


# ------------------------------------------------------ Détection de conflits
def test_detects_vertex_conflict():
    a = Agent(0, (0, 0), (0, 2), path=[(0, 0), (0, 1), (0, 2)])
    b = Agent(1, (1, 1), (0, 1), path=[(1, 1), (0, 1)])
    kinds = {c.kind for c in detect_conflicts([a, b])}
    assert "vertex" in kinds


def test_detects_edge_conflict():
    a = Agent(0, (0, 0), (0, 1), path=[(0, 0), (0, 1)])
    b = Agent(1, (0, 1), (0, 0), path=[(0, 1), (0, 0)])
    conflicts = detect_conflicts([a, b])
    assert len(conflicts) == 1 and conflicts[0].kind == "edge" and conflicts[0].time == 0


def test_no_conflict_for_disjoint_paths():
    a = Agent(0, (0, 0), (0, 2), path=[(0, 0), (0, 1), (0, 2)])
    b = Agent(1, (2, 0), (2, 2), path=[(2, 0), (2, 1), (2, 2)])
    assert detect_conflicts([a, b]) == []


# ------------------------------------------------------------- Coordination
def test_space_time_astar_waits_for_reserved_cell():
    table = ReservationTable()
    table.reserve_path([(0, 1), (1, 1), (2, 1)])                       # occupe (1,1) à t=1
    path = SpaceTimeAStar(Grid(3, 3)).plan((1, 0), (1, 2), table)
    assert path[1] != (1, 1)                                           # n'entre pas en (1,1) à t=1
    assert path[-1] == (1, 2)


def test_prioritized_resolves_crossing():
    grid = Grid(3, 3)
    agents = [Agent(0, (1, 0), (1, 2)), Agent(1, (0, 1), (2, 1))]
    IndependentPlanner(grid).solve(agents)
    assert len(detect_conflicts(agents)) > 0                           # baseline : collision
    PrioritizedPlanner(grid).solve(agents)
    assert detect_conflicts(agents) == []                              # coordination : aucune
    assert all(a.reached_goal for a in agents)


@pytest.mark.parametrize("n_agents", [2, 5])
def test_prioritized_is_conflict_free_on_random_scenarios(n_agents):
    for seed in range(20):
        grid, agents = ScenarioGenerator(10, 10, 0.15, seed=seed).generate(n_agents)
        PrioritizedPlanner(grid).solve(agents)
        assert detect_conflicts(agents) == []


# ---------------------------------------------------------------- Métriques
def test_metrics_and_simulation_do_not_modify_input():
    grid, agents = ScenarioGenerator(10, 10, 0.15, seed=1).generate(5)
    m = Simulation(grid, agents, PrioritizedPlanner).run()
    assert all(a.path == [] for a in agents)                           # copie profonde respectée
    assert m.n_agents == 5 and m.n_conflicts == 0 and m.success_rate == 100.0
    assert m.total_distance >= sum(a.manhattan_distance for a in agents)
    assert m.makespan >= max(a.manhattan_distance for a in agents)
