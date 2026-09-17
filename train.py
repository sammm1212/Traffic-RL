import gymnasium as gym
import traci

from src.simulation.traffic import TrafficSimulation

simulation = TrafficSimulation(traci)

env = gym.make(
    simulation=simulation,
    decision_interval=5,
)

observation, info = env.reset()
