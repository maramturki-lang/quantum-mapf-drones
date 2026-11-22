"""Visualisation : grille, chemins planifiés, conflits et animation."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")                     # rendu sans fenêtre (fichiers PNG/GIF)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .coordinator import detect_conflicts
from .environment import Agent, Grid

COLORS = plt.cm.tab10.colors


def _draw_grid(ax, grid: Grid) -> None:
    ax.set_xlim(-0.5, grid.width - 0.5)
    ax.set_ylim(grid.height - 0.5, -0.5)          # ligne 0 en haut
    ax.set_xticks(range(grid.width))
    ax.set_yticks(range(grid.height))
    ax.grid(True, color="#dddddd", linewidth=0.5)
    ax.set_aspect("equal")
    for r, c in grid.obstacles:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#333333"))


def plot_paths(grid: Grid, agents: list[Agent], ax=None, title: str = ""):
    """Départs (o), cibles (*), chemins et conflits (X rouge)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    _draw_grid(ax, grid)
    for a in agents:
        col = COLORS[a.agent_id % 10]
        if a.path:
            rows, cols = zip(*a.path)
            ax.plot(cols, rows, "-", color=col, linewidth=2, alpha=0.7)
        ax.plot(a.start[1], a.start[0], "o", color=col, markersize=12, label=f"Agent {a.agent_id}")
        ax.plot(a.goal[1], a.goal[0], "*", color=col, markersize=16, markeredgecolor="black")
    for c in detect_conflicts(agents):
        loc = c.location if c.kind == "vertex" else c.location[0]
        ax.plot(loc[1], loc[0], "X", color="red", markersize=14)
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    return ax


def animate(grid: Grid, agents: list[Agent], filename: str, title: str = "", fps: int = 3) -> None:
    """GIF du déplacement des agents pas à pas."""
    horizon = max(len(a.path) for a in agents)
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(t):
        ax.clear()
        _draw_grid(ax, grid)
        for a in agents:
            col = COLORS[a.agent_id % 10]
            ax.plot(a.goal[1], a.goal[0], "*", color=col, markersize=16, markeredgecolor="black")
            r, c = a.position_at(t)
            ax.plot(c, r, "o", color=col, markersize=14)
            ax.text(c, r, str(a.agent_id), ha="center", va="center", color="white", fontsize=8)
        ax.set_title(f"{title}  —  t = {t}")

    anim = FuncAnimation(fig, update, frames=horizon + 2, interval=1000 // fps)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)

