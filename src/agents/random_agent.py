"""Uniformly random controller used as a simple comparison policy."""


class RandomAgent:
    """Select valid traffic-signal actions without learning from experience."""

    def select_action(self, env):
        """Sample and return one action from the environment's action space."""
        return env.action_space.sample()
