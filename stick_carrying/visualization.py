"""Matplotlib visualisations for the stick-carrying MDP and its solution.

A single :class:`StickVisualizer` knows how to draw the environment, the
convergence curve, the optimal value function, the greedy policy and an animated
roll-out of the two robots carrying the stick to the goal.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter

from .solver import Policy, Simulator
from .stick_world import Action, Orientation, StickCarryingMDP, StickState

# action -> arrow direction in (dx, dy) plot coordinates (y grows downward)
_ARROWS = {
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}

_OBSTACLE_COLOR = "#2f3b52"
_FREE_COLOR = "#eef1f6"
_STICK_COLOR = "#8c5a2b"
_ROBOT_A = "#1f77b4"
_ROBOT_B = "#d62728"
_GOAL_COLOR = "#ffce00"


class StickVisualizer:
    def __init__(self, mdp: StickCarryingMDP):
        self.mdp = mdp
        self.size = mdp.world.size
        self._cmap = ListedColormap([_OBSTACLE_COLOR, _FREE_COLOR])

    # --------------------------------------------------------------- helpers
    def _new_ax(self, ax, title):
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(title)
        return ax

    def _draw_grid(self, ax):
        ax.imshow(self.mdp.world.grid, cmap=self._cmap, vmin=0, vmax=1, origin="upper")
        ax.set_xticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.grid(which="minor", color="#9aa4b2", linewidth=0.5)
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.tick_params(length=0)

    def _draw_goal(self, ax):
        gr, gc = self.mdp.goal_cell
        ax.scatter([gc], [gr], marker="*", s=520, color=_GOAL_COLOR,
                   edgecolor="black", linewidth=1.2, zorder=10, label="goal")

    def _draw_stick(self, ax, state: StickState, alpha=1.0, lw=6, label=False):
        (ar, ac), (br, bc) = state.cells()
        ax.plot([ac, bc], [ar, br], color=_STICK_COLOR, lw=lw, alpha=alpha,
                solid_capstyle="round", zorder=5)
        ax.scatter([ac], [ar], s=160, color=_ROBOT_A, edgecolor="black",
                   zorder=6, alpha=alpha, label="robot A" if label else None)
        ax.scatter([bc], [br], s=160, color=_ROBOT_B, edgecolor="black",
                   zorder=6, alpha=alpha, label="robot B" if label else None)

    # ----------------------------------------------------------- environment
    def plot_environment(self, ax=None, show_start=True):
        ax = self._new_ax(ax, "Stick-carrying environment")
        self._draw_grid(ax)
        self._draw_goal(ax)
        if show_start:
            self._draw_stick(ax, self.mdp.start, label=True)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.95)
        return ax

    # ----------------------------------------------------------- convergence
    def plot_convergence(self, history: Sequence[float], ax=None):
        ax = self._new_ax(ax, "Value-iteration convergence")
        ax.plot(range(1, len(history) + 1), history, marker="o", ms=3, color="#2a6f97")
        ax.set_yscale("log")
        ax.set_xlabel("sweep")
        ax.set_ylabel(r"max $|V_{k+1}-V_k|$  (log)")
        ax.grid(True, alpha=0.3)
        return ax

    # ------------------------------------------------------------- value map
    def _value_grid(self, policy: Policy):
        vmap = np.full((self.size, self.size), np.nan)
        for r in range(self.size):
            for c in range(self.size):
                vals = [policy.value(StickState(r, c, o))
                        for o in Orientation if StickState(r, c, o) in policy.values]
                if vals:
                    vmap[r, c] = max(vals)
        return vmap

    def plot_value_function(self, policy: Policy, ax=None, intended_path=None):
        ax = self._new_ax(ax, "Optimal value function $V^*$")
        vmap = np.ma.masked_invalid(self._value_grid(policy))
        im = ax.imshow(vmap, cmap="viridis", origin="upper")
        ax.imshow(np.where(self.mdp.world.grid == 0, 1, np.nan), cmap=self._cmap,
                  vmin=0, vmax=1, origin="upper")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="$V^*$ (anchor cell)")
        if intended_path:
            ys = [s.row for s in intended_path]
            xs = [s.col for s in intended_path]
            ax.plot(xs, ys, color="white", lw=2, alpha=0.9, zorder=4)
            ax.scatter(xs[0], ys[0], color="white", s=40, zorder=5)
        self._draw_goal(ax)
        self._grid_lines(ax)
        return ax

    def _grid_lines(self, ax):
        ax.set_xticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.grid(which="minor", color="#9aa4b2", linewidth=0.4)
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.tick_params(length=0)

    # ---------------------------------------------------------------- policy
    def plot_policy(self, policy: Policy, ax=None):
        ax = self._new_ax(ax, "Greedy policy $\\pi^*$  (arrow = translate, o = rotate)")
        self._draw_grid(ax)
        xs, ys, us, vs = [], [], [], []
        rot_x, rot_y = [], []
        for r in range(self.size):
            for c in range(self.size):
                candidates = [StickState(r, c, o) for o in Orientation
                              if StickState(r, c, o) in policy.values]
                if not candidates:
                    continue
                best = max(candidates, key=policy.value)
                if self.mdp.is_terminal(best):
                    continue
                action = policy.action(best)
                if action == Action.ROTATE:
                    rot_x.append(c)
                    rot_y.append(r)
                elif action in _ARROWS:
                    dx, dy = _ARROWS[action]
                    xs.append(c)
                    ys.append(r)
                    us.append(dx * 0.6)
                    vs.append(dy * 0.6)
        ax.quiver(xs, ys, us, vs, angles="xy", scale_units="xy", scale=1,
                  color="#11324d", width=0.006, zorder=7)
        ax.scatter(rot_x, rot_y, marker="o", facecolors="none",
                   edgecolors="#b5179e", s=90, linewidths=1.8, zorder=7)
        self._draw_goal(ax)
        return ax

    # ------------------------------------------------------------- animation
    def animate(self, trajectory: List[StickState], filename: str = "outputs/rollout.gif",
                fps: int = 3):
        fig, ax = plt.subplots(figsize=(6, 6))

        def draw_frame(i: int):
            ax.clear()
            self._draw_grid(ax)
            self._draw_goal(ax)
            for s in trajectory[: i + 1]:  # faint trail
                ar, ac = s.cells()[0]
                ax.scatter([ac], [ar], s=18, color="#9aa4b2", zorder=3)
            self._draw_stick(ax, trajectory[i])
            reached = self.mdp.is_terminal(trajectory[i])
            tag = "  GOAL!" if reached else ""
            ax.set_title(f"Roll-out  step {i}/{len(trajectory) - 1}{tag}")

        ani = FuncAnimation(fig, draw_frame, frames=len(trajectory), interval=1000 // fps)
        ani.save(filename, writer=PillowWriter(fps=fps))
        plt.close(fig)
        return filename

    def plot_trajectory_snapshots(self, trajectory: List[StickState], k: int = 6,
                                  fig=None):
        idx = np.linspace(0, len(trajectory) - 1, k).round().astype(int)
        fig, axes = plt.subplots(1, k, figsize=(3 * k, 3.4))
        for ax, i in zip(np.atleast_1d(axes), idx):
            self._draw_grid(ax)
            self._draw_goal(ax)
            self._draw_stick(ax, trajectory[i])
            ax.set_title(f"step {i}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("Stochastic roll-out of the optimal policy", y=1.02)
        fig.tight_layout()
        return fig


def dashboard(mdp: StickCarryingMDP, policy: Policy, history: Sequence[float],
              simulator: Optional[Simulator] = None):
    """Convenience 2x2 figure: environment, convergence, value map, policy."""
    viz = StickVisualizer(mdp)
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    viz.plot_environment(ax=axes[0, 0])
    viz.plot_convergence(history, ax=axes[0, 1])
    intended = simulator.intended_path(mdp.start) if simulator else None
    viz.plot_value_function(policy, ax=axes[1, 0], intended_path=intended)
    viz.plot_policy(policy, ax=axes[1, 1])
    fig.tight_layout()
    return fig
