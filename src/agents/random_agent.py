class RandomAgent:
    def select_action(self, env):
        return env.action_space.sample()

