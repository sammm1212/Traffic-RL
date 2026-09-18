"""Deep Q-Network model and epsilon-greedy learning agent.

The agent maps the five-value traffic observation to one estimated return per
signal action. It learns from replay-buffer batches using a separate target
network to provide more stable temporal-difference targets.
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


STATE_SIZE = 5
ACTION_SIZE = 2
HIDDEN_SIZE = 64

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

GAMMA = 0.99
LEARNING_RATE = 1e-3


class DQN(nn.Module):
    """Feed-forward network that estimates Q-values for both signal actions."""

    def __init__(self):
        """Build two hidden ReLU layers and a two-value action output layer."""
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(STATE_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, ACTION_SIZE),
        )

    def forward(self, x):
        """Return estimated action values for a batch of traffic states."""
        return self.network(x)


class DQNAgent:
    """Choose actions with epsilon-greedy exploration and train a DQN."""

    def __init__(self):
        """Create synchronized policy and target networks plus an optimizer."""
        self.policy_network = DQN()
        self.target_network = DQN()

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=LEARNING_RATE,
        )

        self.loss_fn = nn.SmoothL1Loss()
        self.epsilon = EPSILON_START

    def select_action(self, state):
        """Select a random action with probability epsilon, otherwise the best Q-action."""
        if random.random() < self.epsilon:
            return random.randrange(ACTION_SIZE)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_network(state_tensor)

        return q_values.argmax(dim=1).item()

    def decay_epsilon(self):
        """Reduce exploration multiplicatively without dropping below its floor."""
        self.epsilon = max(
            EPSILON_END,
            self.epsilon * EPSILON_DECAY,
        )

    def train_step(self, batch):
        """Update the policy network from one batch of replay transitions.

        The target is the immediate reward plus the discounted value of the
        target network's best next action. Terminal transitions multiply that
        future value by zero because an ended episode has no successor reward.
        """
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.policy_network(states)

        # Only the value of the action actually taken is trained for each row.
        current_q_values = q_values.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(dim=1).values

            target_q_values = (
                rewards
                + GAMMA
                * max_next_q_values
                * (1 - dones)
            )

        loss = self.loss_fn(
            current_q_values,
            target_q_values,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Synchronize the stable target network with current policy weights."""
        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )
