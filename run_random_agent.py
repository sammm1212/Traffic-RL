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
    traci.close()