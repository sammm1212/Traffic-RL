"""Run one headless episode with uniformly random traffic-signal actions.

This script is a simple end-to-end demonstration of the Gymnasium environment:
TraCI starts SUMO, the random agent samples an action every decision interval,
and final reward and vehicle counts are printed when the episode is truncated.
"""

import traci

from src.agents.random_agent import RandomAgent
from src.environment.traffic_env import TrafficEnvironment
from src.simulation.traffic import TrafficSimulation
from src.simulation.run import build_sumo_command, DEFAULT_SEED


sumo_command = build_sumo_command(
    gui=False,
    seed=DEFAULT_SEED,
    steps=300,
)

traci.start(sumo_command)

try:
    simulation = TrafficSimulation(
        traci,
        reload_args=sumo_command[1:],
    )

    env = TrafficEnvironment(
        simulation=simulation,
        decision_interval=5,
    )

    agent = RandomAgent()

    observation, info = env.reset()

    total_reward = 0.0
    step_count = 0

    while True:
        # The random policy delegates sampling to the environment so every
        # selected value belongs to its declared discrete action space.
        action = agent.select_action(env)

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        metrics = simulation.observe()

        if terminated or truncated:
            break

    print("Total reward:", total_reward)
    print("Total steps:", step_count)
    print('Final observation:', observation)
    print("Vehicles completed:", metrics.vehicles_completed)
    print("Vehicles generated:", metrics.vehicles_generated)

finally:
    # Closing TraCI also terminates the SUMO child process and releases its port.
    traci.close()
