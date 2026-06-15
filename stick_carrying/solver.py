"""Planning and simulation for any :class:`~stick_carrying.mdp.MDP`.

``ValueIterationSolver`` performs the classic Bellman-optimality dynamic program
and returns a :class:`Policy`.  ``Simulator`` rolls a policy out through the
stochastic environment to produce a trajectory for visualisation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Optional

import numpy as np

from .mdp import MDP, A, S


class Policy(Generic[S, A]):
    """A greedy policy together with the optimal state-value function."""

    def __init__(self, values: Dict[S, float], actions: Dict[S, Optional[A]]):
        self.values = values
        self._actions = actions

    def value(self, state: S) -> float:
        return self.values[state]

    def action(self, state: S) -> Optional[A]:
        """Best action in ``state`` (``None`` for terminal states)."""
        return self._actions[state]


class Solver(ABC, Generic[S, A]):
    """Anything that turns an MDP into a :class:`Policy`."""

    def __init__(self, mdp: MDP[S, A]):
        self.mdp = mdp

    @abstractmethod
    def solve(self) -> Policy[S, A]:
        ...

    def q_value(self, values: Dict[S, float], state: S, action: A) -> float:
        """Expected one-step-lookahead value of ``action`` under ``values``."""
        mdp, gamma = self.mdp, self.mdp.discount
        q = 0.0
        for nxt, prob in mdp.transitions(state, action).items():
            cont = 0.0 if mdp.is_terminal(nxt) else values[nxt]
            q += prob * (mdp.reward(state, action, nxt) + gamma * cont)
        return q


class ValueIterationSolver(Solver[S, A]):
    """Synchronous value iteration.

    Repeatedly applies the Bellman optimality backup

        V(s) <- max_a  sum_s' P(s'|s,a) [ R(s,a,s') + gamma * V(s') ]

    until the largest update falls below ``theta``.  The per-sweep maximum
    change is recorded in :attr:`history` for the convergence plot.
    """

    def __init__(self, mdp: MDP[S, A], theta: float = 1e-4, max_iter: int = 1000):
        super().__init__(mdp)
        self.theta = theta
        self.max_iter = max_iter
        self.history: List[float] = []

    def solve(self) -> Policy[S, A]:
        mdp = self.mdp
        states = list(mdp.states())
        values: Dict[S, float] = {s: 0.0 for s in states}
        self.history = []

        for _ in range(self.max_iter):
            delta = 0.0
            updated: Dict[S, float] = {}
            for s in states:
                if mdp.is_terminal(s):
                    updated[s] = 0.0
                    continue
                best = max(self.q_value(values, s, a) for a in mdp.actions(s))
                updated[s] = best
                delta = max(delta, abs(best - values[s]))
            values = updated
            self.history.append(delta)
            if delta < self.theta:
                break

        return Policy(values, self._greedy_actions(values))

    def _greedy_actions(self, values: Dict[S, float]) -> Dict[S, Optional[A]]:
        mdp = self.mdp
        policy: Dict[S, Optional[A]] = {}
        for s in mdp.states():
            if mdp.is_terminal(s):
                policy[s] = None
                continue
            policy[s] = max(mdp.actions(s), key=lambda a: self.q_value(values, s, a))
        return policy


class Simulator(Generic[S, A]):
    """Roll a policy out through the (stochastic) environment."""

    def __init__(self, mdp: MDP[S, A], policy: Policy[S, A]):
        self.mdp = mdp
        self.policy = policy

    def run(self, start: S, rng: Optional[np.random.Generator] = None,
            max_steps: int = 200) -> List[S]:
        """Return the sampled trajectory of states from ``start``."""
        rng = rng or np.random.default_rng()
        state = start
        trajectory = [state]
        for _ in range(max_steps):
            if self.mdp.is_terminal(state):
                break
            action = self.policy.action(state)
            assert action is not None
            dist = self.mdp.transitions(state, action)
            outcomes, probs = zip(*dist.items())
            state = outcomes[rng.choice(len(outcomes), p=np.asarray(probs))]
            trajectory.append(state)
        return trajectory

    def intended_path(self, start: S, max_steps: int = 200) -> List[S]:
        """The deterministic path the robots *intend* to follow (ignoring slips).

        At each step we take the most-likely successor under the optimal action,
        which is exactly the manoeuvre the robots aim to perform.
        """
        state = start
        path = [state]
        seen = {state}
        for _ in range(max_steps):
            if self.mdp.is_terminal(state):
                break
            action = self.policy.action(state)
            assert action is not None
            nxt = max(self.mdp.transitions(state, action).items(), key=lambda kv: kv[1])[0]
            path.append(nxt)
            if nxt == state or nxt in seen:
                break
            seen.add(nxt)
            state = nxt
        return path
