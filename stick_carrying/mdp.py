"""Abstract Markov Decision Process interface.

The whole project is built around this small abstract base class.  Anything that
can enumerate its states/actions, expose a (possibly stochastic) transition
model and hand out rewards can be plugged into the solvers in ``solver.py``.
The concrete stick-carrying problem lives in ``stick_world.py``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Generic, Hashable, Iterable, Sequence, TypeVar

S = TypeVar("S", bound=Hashable)  # state type
A = TypeVar("A", bound=Hashable)  # action type


class MDP(ABC, Generic[S, A]):
    """A discrete Markov Decision Process, generic over state and action types.

    Sub-classes describe a problem by implementing five methods plus the
    ``discount`` property.  Transitions are returned as ``{next_state: prob}``
    dictionaries, which keeps the solvers completely agnostic to the underlying
    domain.
    """

    # ------------------------------------------------------------------ core
    @property
    @abstractmethod
    def discount(self) -> float:
        """Discount factor gamma in [0, 1)."""

    @abstractmethod
    def states(self) -> Sequence[S]:
        """Return every reachable state of the problem."""

    @abstractmethod
    def actions(self, state: S) -> Iterable[A]:
        """Return the actions available in ``state``."""

    @abstractmethod
    def transitions(self, state: S, action: A) -> Dict[S, float]:
        """Return ``{next_state: probability}`` for taking ``action`` in ``state``."""

    @abstractmethod
    def reward(self, state: S, action: A, next_state: S) -> float:
        """Immediate reward for the transition ``state -> next_state``."""

    @abstractmethod
    def is_terminal(self, state: S) -> bool:
        """Whether ``state`` is absorbing (the episode ends here)."""

    # -------------------------------------------------------------- utilities
    def reachable_to_terminal(self) -> set:
        """Backward breadth-first search returning every state from which some
        terminal state is reachable.  Used to guarantee a solvable instance."""
        states = list(self.states())
        index = set(states)
        predecessors: Dict[S, set] = {s: set() for s in states}

        for s in states:
            if self.is_terminal(s):
                continue
            successors: set = set()
            for a in self.actions(s):
                successors.update(self.transitions(s, a).keys())
            for nxt in successors:
                if nxt in index:
                    predecessors[nxt].add(s)

        frontier = deque(s for s in states if self.is_terminal(s))
        seen = set(frontier)
        while frontier:
            current = frontier.popleft()
            for pred in predecessors[current]:
                if pred not in seen:
                    seen.add(pred)
                    frontier.append(pred)
        return seen

