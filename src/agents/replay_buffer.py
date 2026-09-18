"""Bounded replay memory for DQN state transitions."""

from collections import deque
import random


class ReplayBuffer:
    """Retain recent experiences and sample uncorrelated training batches."""

    def __init__(self, capacity):
        """Create a FIFO buffer that automatically discards its oldest entries."""
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store one state-action transition and its episode-end marker."""
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        """Return a uniformly sampled batch without replacement."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Return the number of transitions currently available."""
        return len(self.buffer)
