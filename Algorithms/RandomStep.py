import numpy as np


class RandomStep():
    def __init__(self, num_actions: int = 4, num_agents: int = 2) -> None:
        self.num_actions = num_actions
        self.num_agents = num_agents
        
    def get_action(self, state):
        return np.random.randint(0, self.num_actions)