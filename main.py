"""End-to-end driver for the multi-robot stick-carrying problem.

Builds a solvable stochastic grid, solves it with value iteration, simulates a
roll-out of the optimal policy and writes every figure to ``outputs/``.

Run::

    python main.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless: just write files
import matplotlib.pyplot as plt
import numpy as np

from stick_carrying import (
    Orientation,
    Simulator,
    StickCarryingMDP,
    StickState,
    StickVisualizer,
    ValueIterationSolver,
    dashboard,
)

OUT = "outputs"
SEED = 7


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # 1. Build a solvable stochastic environment ---------------------------
    start = StickState(0, 0, Orientation.HORIZONTAL)
    mdp = StickCarryingMDP.random_solvable(
        size=10, start=start, goal_cell=(9, 9), rng=rng,
        p_main=0.8, discount=0.95,
    )
    print(f"[env]    grid generated on attempt {mdp.generation_attempts}, "
          f"{len(mdp.states())} valid stick states")

    # 2. Plan with value iteration -----------------------------------------
    solver = ValueIterationSolver(mdp, theta=1e-4)
    policy = solver.solve()
    print(f"[solve]  value iteration converged in {len(solver.history)} sweeps "
          f"(final delta = {solver.history[-1]:.2e})")
    print(f"[solve]  V*(start) = {policy.value(start):.2f}")

    # 3. Simulate the stochastic roll-out ----------------------------------
    sim = Simulator(mdp, policy)
    trajectory = sim.run(start, rng=np.random.default_rng(3))
    reached = mdp.is_terminal(trajectory[-1])
    print(f"[sim]    roll-out reached goal in {len(trajectory) - 1} steps "
          f"(success={reached})")

    # 4. Visualise ---------------------------------------------------------
    viz = StickVisualizer(mdp)

    fig = dashboard(mdp, policy, solver.history, simulator=sim)
    fig.savefig(f"{OUT}/dashboard.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    snap = viz.plot_trajectory_snapshots(trajectory, k=6)
    snap.savefig(f"{OUT}/trajectory.png", dpi=130, bbox_inches="tight")
    plt.close(snap)

    gif = viz.animate(trajectory, filename=f"{OUT}/rollout.gif", fps=3)

    print(f"[viz]    wrote {OUT}/dashboard.png, {OUT}/trajectory.png, {gif}")


if __name__ == "__main__":
    main()
