"""The multi-robot stick-carrying problem modelled as a stochastic MDP.

Two robots rigidly hold the two ends of a stick of length one.  Together they
occupy two neighbouring grid cells.  The *configuration* of the stick is the
state of the MDP::

    StickState(row, col, orientation)

where ``(row, col)`` is the cell of robot A and the orientation decides where
robot B sits (one cell to the right for HORIZONTAL, one cell below for
VERTICAL).  The robots translate the stick (up/down/left/right) or rotate it
about robot A.  The environment is *stochastic*: with probability ``p_main`` the
intended manoeuvre succeeds, otherwise the imperfectly-coordinated robots
"slip" and the stick translates in a random direction instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .mdp import MDP

Cell = Tuple[int, int]


class Orientation(IntEnum):
    HORIZONTAL = 0  # robot B sits at (row, col + 1)
    VERTICAL = 1    # robot B sits at (row + 1, col)


class Action(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    ROTATE = "rotate"


#: translation actions and their (drow, dcol) effect
TRANSLATIONS: Dict[Action, Cell] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


@dataclass(frozen=True)
class StickState:
    """Immutable, hashable configuration of the carried stick."""

    row: int
    col: int
    orient: Orientation

    def cells(self) -> Tuple[Cell, Cell]:
        """The two grid cells occupied by robot A and robot B."""
        a = (self.row, self.col)
        if self.orient == Orientation.HORIZONTAL:
            b = (self.row, self.col + 1)
        else:
            b = (self.row + 1, self.col)
        return a, b


class GridWorld:
    """A static occupancy grid: ``1`` = free, ``0`` = obstacle."""

    FREE = 1
    OBSTACLE = 0

    def __init__(self, grid: np.ndarray):
        self.grid = np.asarray(grid, dtype=int)
        self.size = self.grid.shape[0]

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def is_free(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.grid[r, c] == self.FREE

    @classmethod
    def random(
        cls,
        size: int,
        rng: np.random.Generator,
        p_free_range: Tuple[float, float] = (0.78, 0.9),
        free_cells: Sequence[Cell] = (),
    ) -> "GridWorld":
        """Sample a random grid; ``free_cells`` are forced to be obstacle-free."""
        p_free = rng.uniform(*p_free_range)
        grid = rng.choice(
            [cls.OBSTACLE, cls.FREE], p=[1 - p_free, p_free], size=(size, size)
        ).astype(int)
        for (r, c) in free_cells:
            if 0 <= r < size and 0 <= c < size:
                grid[r, c] = cls.FREE
        return cls(grid)


class StickCarryingMDP(MDP[StickState, Action]):
    """Stochastic MDP for cooperatively carrying a stick to a goal cell."""

    def __init__(
        self,
        world: GridWorld,
        start: StickState,
        goal_cell: Cell,
        p_main: float = 0.8,
        discount: float = 0.95,
        step_reward: float = -1.0,
        goal_reward: float = 100.0,
    ):
        self.world = world
        self.start = start
        self.goal_cell = goal_cell
        self.p_main = p_main
        self._discount = discount
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.generation_attempts = 0
        self._states: Optional[List[StickState]] = None

    # ----------------------------------------------------------- MDP interface
    @property
    def discount(self) -> float:
        return self._discount

    def states(self) -> Sequence[StickState]:
        if self._states is None:
            self._states = [s for s in self._enumerate() if self._valid(s)]
        return self._states

    def actions(self, state: StickState) -> Sequence[Action]:
        return list(Action)

    def transitions(self, state: StickState, action: Action) -> Dict[StickState, float]:
        if self.is_terminal(state):
            return {state: 1.0}
        dist: Dict[StickState, float] = {}
        intended = self._apply(state, action)
        dist[intended] = dist.get(intended, 0.0) + self.p_main
        slip = (1.0 - self.p_main) / len(TRANSLATIONS)
        for move in TRANSLATIONS:
            nxt = self._apply(state, move)
            dist[nxt] = dist.get(nxt, 0.0) + slip
        return dist

    def reward(self, state: StickState, action: Action, next_state: StickState) -> float:
        return self.goal_reward if self.is_terminal(next_state) else self.step_reward

    def is_terminal(self, state: StickState) -> bool:
        a, b = state.cells()
        return self.goal_cell == a or self.goal_cell == b

    # ------------------------------------------------------------- mechanics
    def _enumerate(self):
        for r in range(self.world.size):
            for c in range(self.world.size):
                for o in Orientation:
                    yield StickState(r, c, o)

    def _valid(self, state: StickState) -> bool:
        a, b = state.cells()
        return self.world.is_free(*a) and self.world.is_free(*b)

    def _apply(self, state: StickState, action: Action) -> StickState:
        """Deterministic intended outcome; an invalid move leaves the stick put."""
        if action == Action.ROTATE:
            candidate = StickState(state.row, state.col, Orientation(1 - state.orient))
        else:
            dr, dc = TRANSLATIONS[action]
            candidate = StickState(state.row + dr, state.col + dc, state.orient)
        return candidate if self._valid(candidate) else state

    # --------------------------------------------------------------- factory
    @classmethod
    def random_solvable(
        cls,
        size: int = 10,
        start: Optional[StickState] = None,
        goal_cell: Optional[Cell] = None,
        rng: Optional[np.random.Generator] = None,
        max_attempts: int = 500,
        **params,
    ) -> "StickCarryingMDP":
        """Generate random grids until the start can provably reach the goal."""
        rng = rng or np.random.default_rng()
        start = start or StickState(0, 0, Orientation.HORIZONTAL)
        goal_cell = goal_cell or (size - 1, size - 1)
        gr, gc = goal_cell
        forced = list(start.cells()) + [goal_cell, (gr, gc - 1), (gr - 1, gc)]

        for attempt in range(max_attempts):
            world = GridWorld.random(size, rng, free_cells=forced)
            mdp = cls(world, start, goal_cell, **params)
            if not mdp._valid(start):
                continue
            if start in mdp.reachable_to_terminal():
                mdp.generation_attempts = attempt
                return mdp
        raise RuntimeError("Could not generate a solvable grid; relax the parameters.")
