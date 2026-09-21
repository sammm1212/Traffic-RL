"""Gymnasium interface around the reusable TraCI traffic simulation adapter.

An observation contains four approach queue lengths followed by SUMO's current
traffic-light phase index. Actions select the desired north-south or east-west
green, and rewards penalize the total queue observed after each interval.
"""

from collections.abc import Callable

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.simulation.metrics import (
    MetricsAccumulator,
    SimulationSummary,
    TrafficMetrics,
)


MINIMUM_GREEN_DURATION = 10


class TrafficEnvironment(gym.Env):
    """Expose one SUMO-controlled junction through the Gymnasium API."""

    def __init__(
        self,
        simulation,
        decision_interval=5,
        max_simulation_time=300,
        minimum_green_duration: int | None = None,
        step_observer: Callable[[TrafficMetrics], None] | None = None,
    ):
        """Configure action timing and an optional DQN minimum-green guard."""
        super().__init__()

        if decision_interval <= 0:
            raise ValueError("decision_interval must be positive")
        if minimum_green_duration is not None and minimum_green_duration < 0:
            raise ValueError("minimum_green_duration cannot be negative")

        self.simulation = simulation
        self.decision_interval = decision_interval
        self.minimum_green_duration = minimum_green_duration
        self._step_observer = step_observer

        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(5,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)

        self.max_simulation_time = max_simulation_time
        self._episode_metrics = MetricsAccumulator()
        self._active_green_direction = 0
        self._green_elapsed_seconds = 0

    def get_state(self):
        """Return approach queues and signal phase as a float32 observation."""
        metrics = self.simulation.observe()

        state = metrics.state_vector()

        return np.array(state, dtype=np.float32)

    def _get_obs(self):
        """Build the current Gymnasium observation without advancing SUMO."""
        return self.get_state()

    def _get_reward(self):
        """Penalize the total number of halted vehicles on all approaches."""
        metrics = self.simulation.observe()

        total_queue = sum(
            approach.queue_length
            for approach in metrics.approaches.values()
        )

        return -float(total_queue)

    def _record_step(self, metrics: TrafficMetrics) -> None:
        """Record an internal SUMO step and notify an optional observer."""
        self._episode_metrics.add(metrics)
        if self._step_observer is not None:
            self._step_observer(metrics)

    @staticmethod
    def _direction_for_phase(phase: int) -> int:
        """Map a principal SUMO green phase to the two-action interface."""
        if phase == 0:
            return 0
        if phase == 3:
            return 1
        raise RuntimeError(
            f"episode reset did not start on a principal green phase: {phase}"
        )

    def reset(self, *, seed=None, options=None):
        """Reload the configured SUMO episode and return its initial state."""
        super().reset(seed=seed)

        self.simulation.reset()
        self._episode_metrics = MetricsAccumulator()
        self._active_green_direction = self._direction_for_phase(
            self.simulation.observe().light_phase
        )
        self._green_elapsed_seconds = 0

        observation = self._get_obs()
        info = {}

        return observation, info

    def episode_summary(self) -> SimulationSummary:
        """Return traffic aggregates collected during the current episode."""
        return self._episode_metrics.summary()

    def step(self, action):
        """Apply a target green phase for one interval and return the transition."""
        requested_action = int(action)
        if requested_action not in (0, 1):
            raise ValueError(f"Invalid traffic-light action: {requested_action}")

        switch_requested = requested_action != self._active_green_direction
        switch_blocked = (
            switch_requested
            and self.minimum_green_duration is not None
            and self._green_elapsed_seconds < self.minimum_green_duration
        )
        effective_action = (
            self._active_green_direction if switch_blocked else requested_action
        )

        self.simulation.apply_action(
            effective_action,
            self.decision_interval,
            on_step=self._record_step,
        )

        if effective_action == self._active_green_direction:
            self._green_elapsed_seconds += self.decision_interval
        else:
            self._active_green_direction = effective_action
            # Count completed full green-hold intervals after the transition.
            # This conservatively guarantees at least the configured duration.
            self._green_elapsed_seconds = 0

        observation = self._get_obs()
        reward = self._get_reward()
        current_time = self.simulation.observe().time
        terminated = False
        truncated = current_time >= self.max_simulation_time
        info = {
            "requested_action": requested_action,
            "effective_action": effective_action,
            "switch_blocked": switch_blocked,
            "green_elapsed_seconds": self._green_elapsed_seconds,
        }

        return observation, reward, terminated, truncated, info


