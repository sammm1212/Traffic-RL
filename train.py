"""Train the repository's prototype DQN controller against the SUMO environment.

Each environment transition is stored in replay memory. Once enough transitions
exist for a full batch, the policy network learns from randomly sampled past
experience and the target network is periodically synchronized for stability.
"""

import traci

from src.agents.dqn_agent import DQNAgent
from src.agents.replay_buffer import ReplayBuffer
from src.environment.traffic_env import TrafficEnvironment
from src.simulation.traffic import TrafficSimulation
from src.simulation.run import build_sumo_command


NUM_EPISODES = 3
BATCH_SIZE = 64
REPLAY_CAPACITY = 10_000
TARGET_UPDATE_FREQUENCY = 500

EPISODE_SECONDS = 300
SEED = 0

command = build_sumo_command(
    gui=False,
    seed=SEED,
    steps=EPISODE_SECONDS,
)


traci.start(command)

simulation = TrafficSimulation(traci, reload_args=command[1:])

env = TrafficEnvironment(
    simulation=simulation,
    decision_interval=5,
)

agent = DQNAgent()

replay_buffer = ReplayBuffer(
    capacity=REPLAY_CAPACITY,
)

training_step = 0

try:
    for episode in range(NUM_EPISODES):
        state, info = env.reset()

        terminated = False
        truncated = False

        episode_reward = 0.0
        episode_losses = []

        while not (terminated or truncated):
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            # Store complete transitions so training batches are not limited to
            # the strongly correlated sequence most recently produced by SUMO.
            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done,
            )

            state = next_state
            episode_reward += reward

            if len(replay_buffer) >= BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)

                loss = agent.train_step(batch)

                episode_losses.append(loss)

                training_step += 1

                if training_step % TARGET_UPDATE_FREQUENCY == 0:
                    # A fixed target network makes the bootstrapped Q-learning
                    # target change less rapidly than the policy network.
                    agent.update_target_network()

        # Exploration is reduced once per completed episode, down to the floor
        # enforced by DQNAgent.decay_epsilon().
        agent.decay_epsilon()

        if episode_losses:
            mean_loss = sum(episode_losses) / len(episode_losses)
        else:
            mean_loss = 0.0

        print(
            f"Episode {episode + 1}/{NUM_EPISODES} | "
            f"Reward: {episode_reward:.2f} | "
            f"Loss: {mean_loss:.4f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )
finally:
    # Ensure the external SUMO process is not left running after errors or Ctrl-C.
    traci.close()
