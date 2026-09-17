import traci
import gymnasium as gym
from gymnasium import spaces
import numpy as np



class TrafficEnvironment(gym.Env):

    def __init__(self, simulation, decision_interval=5):
        super().__init__()

        self.simulation = simulation
        self.decision_interval = decision_interval

        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(5,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)

        self.max_simulation_time = 300 

    def get_state(self):
        metrics = self.simulation.observe()

        state = metrics.state_vector()

        return np.array(state, dtype=np.float32)

    def _get_obs(self):
        return self.get_state()

    def _get_reward(self):
            metrics = self.simulation.observe()
    
            total_queue = sum(
                approach.queue_length
                for approach in metrics.approaches.values()
            )
    
            return -float(total_queue)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.simulation.reset()

        observation = self._get_obs()
        info = {}

        return observation, info

    def step(self, action):
        self.simulation.apply_action(
        action,
        self.decision_interval,
        )

        observation = self._get_obs()
        reward = self._get_reward()
        current_time = self.simulation.observe().time
        terminated = False
        truncated = current_time >= self.max_simulation_time
        info = {}

        return observation, reward, terminated, truncated, info







