from collections import deque
from copy import deepcopy as dcopy
import random
import numpy as np
from Board.screen import Screen
from src.player import Player
from src.state import State
import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

class AgentFighting(object):
    def __init__(self, args, configs, show_screen = False):
        self.args = args
        self.configs = configs
        self.show_screen = show_screen
        
        self.action_space = {
            'Move': ['U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR'],
            'Change': ['U', 'D', 'L', 'R'],
            'Stay': 1
        }
        
        self.n_actions = len(self.action_space['Move']) + len(self.action_space['Change']) + 1
        self.num_players = 2
        self.screen = Screen(render=self.show_screen)
        self.players = [Player(i, self.num_players) for i in range(self.num_players)]
        self.current_player = 0
        self.state = None
        self.reset()
        
    def render(self):
        if self.show_screen:
            self.screen.load_state(self.state)
            self.screen.render()
    
    def save_image(self, path):
        if self.show_screen:
            self.screen.save(path)
    
    def reset(self):
        """
        Resets the game by resetting player scores, creating a new map, and initializing the game state.
        :return: None
        """
        self.players[0].reset_scores()
        self.players[1].reset_scores()
        self.state = State(self.configs['map'], action_space=self.action_space)
        self.state.set_players(self.players)
        self.num_agents = self.state.num_agents
        self.state.make_random_map()
        if self.show_screen:
            self.screen.init(self.state)
        self.num_agents = self.state.num_agents
        state = self.state.get_state()
        return state
    
    def in_bounds(self, coords):
        return 0 <= coords[0] < self.state.height and 0 <= coords[1] < self.state.width
    
    def is_valid_action(self, action):
        return action < self.n_actions
    
    def get_space_size(self):
        return self.get_state()['observation'].shape
            
    def get_state(self, obj=False):
        if obj:
            return dcopy(self.state)
        else:
            return self.state.get_state()
    
    def is_terminal(self):
        """
        Checks if the game has ended by evaluating if there are any remaining turns left.

        :return: Boolean value indicating whether or not the game has ended.
        """
        return self.state.is_terminal()
            
    def get_winner(self):
        """
        Returns the winner of the game based on the scores of the two players.
        :return: 1 if player 1 wins, -1 if player 2 wins, and 0 if it's a tie.
        """
        if self.state.scores[0] > self.state.scores[1]:
            return 1
        elif self.state.scores[1] > self.state.scores[0]:
            return -1
        else:
            return 0
    
    def flip(self, matrix):
        return np.flip(matrix, axis=1)
    
    def rotate(self, matrix, k=1):
        return np.rot90(matrix, k=k)

    def get_symmetry_transition(self, state, action, next_state):
        flip = random.choice([True, False])
        action_type = self.state.get_type_action(action)
        if action_type[0] == 'Stay':
            return state, action, next_state
        
        if flip:
            direction = action_type[1]
            if action_type[0] == 'Move' or action_type[0] == 'Change':
                if direction == 'L':
                    direction = 'R'
                elif direction == 'R':
                    direction = 'L'
                elif direction == 'UL':
                    direction = 'UR'
                elif direction == 'UR':
                    direction = 'UL'
                elif direction == 'DL':
                    direction = 'DR'
                elif direction == 'DR':
                    direction = 'DL'
            
            action = self.state.action_map[(action_type[0], direction)]
                
            for i in range(state.shape[0]):
                state_layer = self.flip(state[i])
                state[i] = state_layer
                next_state_layer = self.flip(next_state[i])
                next_state[i] = next_state_layer
                
        action_type = self.state.get_type_action(action)
        k = random.choice([0, 1, 2, 3])
        
        for i in range(state.shape[0]):
            state_layer = self.rotate(state[i], k=k)
            state[i] = state_layer
            next_state_layer = self.rotate(next_state[i], k=k)
            next_state[i] = next_state_layer
            
        if action_type[0] == 'Move' or action_type[0] == 'Change':
            direction = action_type[1]
            for i in range(k):
                if direction == 'L':
                    direction = 'D'
                elif direction == 'R':
                    direction = 'U'
                elif direction == 'D':
                    direction = 'R'
                elif direction == 'U':
                    direction = 'L'
                elif direction == 'UL':
                    direction = 'DL'
                elif direction == 'DL':
                    direction = 'DR'
                elif direction == 'DR':
                    direction = 'UR'
                elif direction == 'UR':
                    direction = 'UL'
                    
                action = self.state.action_map[(action_type[0], direction)]
                    
            action = self.state.action_map[(action_type[0], direction)]
    
        return state, action, next_state
    
    def get_valid_actions(self, state=None):
        valids = np.zeros(self.n_actions, dtype=bool)
        
        for action in range(self.n_actions):
            valids[action] = self.is_valid_action(action)
            
        return valids
                    
    def step(self, action, verbose=False):
        """
        This function performs a single step of the game by taking an action as input. The action 
        should be valid or else the function returns the reward. If the action is valid, then the 
        function updates the state of the game and returns the reward.

        Args:
            action: The action to be taken in the game.

        Returns:
            reward: The reward obtained from the step.
        """
        current_player = self.state.get_curr_player()
        is_valid_action = self.state.next(action)
            
        if self.show_screen:
            self.screen.load_state(self.state)
            self.screen.render()
        
        if is_valid_action:
            new_scores = self.state.scores
            if new_scores[current_player] > new_scores[1 - current_player]:
                reward = 1
            elif new_scores[current_player] < new_scores[1 - current_player]:
                reward = -1
            else:
                reward = 0
        else:
            reward = -1
            
        return self.get_state(), reward, self.is_terminal()