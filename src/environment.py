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
        
        self.action_list = {
            'Move': ['U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR'],
            'Build': ['U', 'D', 'L', 'R'],
            'Destroy': ['U', 'D', 'L', 'R'],
            'Stay': 1
        }
        
        self.action_map = {
            ('Move', 'U'): 0,
            ('Move', 'D'): 1,
            ('Move', 'L'): 2,
            ('Move', 'R'): 3,
            ('Move', 'UL'): 4,
            ('Move', 'UR'): 5,
            ('Move', 'DL'): 6,
            ('Move', 'DR'): 7,
            ('Build', 'U'): 8,
            ('Build', 'D'): 9,
            ('Build', 'L'): 10,
            ('Build', 'R'): 11,
            ('Destroy', 'U'): 12,
            ('Destroy', 'D'): 13,
            ('Destroy', 'L'): 14,
            ('Destroy', 'R'): 15,
            ('Stay', 'Stay'): 16
        }
        
        self.direction_map = {
            'U': (-1, 0),
            'D': (1, 0),
            'L': (0, -1),
            'R': (0, 1),
            'UL': (-1, -1),
            'UR': (-1, 1),
            'DL': (1, -1),
            'DR': (1, 1)
        }
        
        self.n_actions = len(self.action_list['Move']) + len(self.action_list['Build']) + len(self.action_list['Destroy']) + 1
        self.num_players = 2
        self.screen = Screen(self)
        self.players = [Player(i, self.num_players) for i in range(self.num_players)]
        self.current_player = 0
        self.history_size = 10
        self.history = deque(maxlen=self.history_size)
        self.reset()
        
    def render(self):
        if self.show_screen:
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
        self.state = State(self.configs['map'])
        self.state.set_players(self.players)
        self.num_agents = self.state.num_agents
        self.state.make_random_map()
        self.screen.init(self.state)
        self.num_agents = self.state.num_agents
        state = self.state.get_state()
        self.history.append(state)
        return state
    
    def in_bounds(self, coords):
        return 0 <= coords[0] < self.state.height and 0 <= coords[1] < self.state.width
    
    def is_valid_action(self, action):
        return action < self.n_actions
    
    def get_type_action(self, action):
        """
        Returns the type of the given action and the corresponding action list item.

        :param action: An integer representing the index of the action in the flattened action list.
        :return: A tuple of strings. The first string is the type of the action ('Move', 'Build', 'Destroy', or 'Stay').
                The second string is the corresponding item from the action list.
        """
        move_len = len(self.action_list['Move'])
        build_len = len(self.action_list['Build'])
        destroy_len = len(self.action_list['Destroy'])

        if action < move_len:
            return ('Move', self.action_list['Move'][action])
        elif action < move_len + build_len:
            return ('Build', self.action_list['Build'][action - move_len])
        elif action < move_len + build_len + destroy_len:
            return ('Destroy', self.action_list['Destroy'][action - move_len - build_len])
        else:
            return ('Stay',)

    
    def get_space_size(self):
        return self.get_state()['observation'].shape
            
    def get_state(self):
        return self.state.get_state()
    
    def game_ended(self):
        """
        Checks if the game has ended by evaluating if there are any remaining turns left.

        :return: Boolean value indicating whether or not the game has ended.
        """
        return self.state.remaining_turns == 0
            
    def get_winner(self):
        """
        Returns the winner of the game.

        :return: An integer representing the winner of the game.
        """
        if self.state.scores[0] > self.state.scores[1]:
            return 0
        elif self.state.scores[1] > self.state.scores[0]:
            return 1
        else:
            return -1
    
    def is_valid_action(self, action):
        current_player = self.state.current_player
        agent_coords_in_order = self.state.agent_coords_in_order
        agent_current_idx = self.state.agent_current_idx
        current_position = agent_coords_in_order[current_player][agent_current_idx]
        
        is_valid_action = True
        action_type = self.get_type_action(action)
        if action_type[0] == 'Move':
            direction = action_type[1]
            next_position = (self.direction_map[direction][0] + current_position[0],
                        self.direction_map[direction][1] + current_position[1])
            if not self.in_bounds(next_position):
                is_valid_action = False
                
            elif next_position in self.state.agent_coords_in_order[0] or \
                        next_position in self.state.agent_coords_in_order[1]:
                is_valid_action = False
                
            elif self.state.agents[current_player][next_position[0]][next_position[1]] == 1:
                ''' in turn (N agent actions at the same time), only one agent can move at an area, 
                    so the other agent is moved into the same area befores
                    agents save next coordinates but agent_coords_in_order is not updated to check this '''
                is_valid_action = False
                
            elif self.state.walls[0][next_position[0]][next_position[1]] == 1 \
                    or self.state.walls[1][next_position[0]][next_position[1]] == 1:
                is_valid_action = False
                
            elif self.state.castles[next_position[0]][next_position[1]] == 1 \
                    or self.state.castles[next_position[0]][next_position[1]] == 1:
                is_valid_action = False
    
        elif action_type[0] == 'Build':
            direction = action_type[1]
            wall_coord = (self.direction_map[direction][0] + current_position[0],
                        self.direction_map[direction][1] + current_position[1])
            if not self.in_bounds(wall_coord):
                is_valid_action = False
                
            elif self.state.walls[0][wall_coord[0]][wall_coord[1]] == 1 \
                    or self.state.walls[1][wall_coord[0]][wall_coord[1]] == 1:
                is_valid_action = False
                
            elif self.state.castles[wall_coord[0]][wall_coord[1]] == 1 \
                    or self.state.castles[wall_coord[0]][wall_coord[1]] == 1:
                is_valid_action = False
                
            elif wall_coord in self.state.agent_coords_in_order[0] or \
                        wall_coord in self.state.agent_coords_in_order[1]:
                is_valid_action = False
        elif action_type[0] == 'Destroy':
            direction = action_type[1]
            wall_coord = (self.direction_map[direction][0] + current_position[0],
                        self.direction_map[direction][1] + current_position[1])
            if not self.in_bounds(wall_coord):
                is_valid_action = False
                
            elif self.state.walls[0][wall_coord[0]][wall_coord[1]] == 0 \
                    and self.state.walls[1][wall_coord[0]][wall_coord[1]] == 0:
                is_valid_action = False
                
        return is_valid_action
    
    def flip(self, matrix):
        return np.flip(matrix, axis=1)
    
    def rotate(self, matrix, k=1):
        return np.rot90(matrix, k=k)

    def get_symmetry_transition(self, state, action, next_state):
        flip = random.choice([True, False])
        action_type = self.get_type_action(action)
        if action_type[0] == 'Stay':
            return state, action, next_state
        
        if flip:
            direction = action_type[1]
            if action_type[0] == 'Move' or action_type[0] == 'Build' or action_type[0] == 'Destroy':
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
            
            action = self.action_map[(action_type[0], direction)]
                
            for i in range(state.shape[0]):
                state_layer = self.flip(state[i])
                state[i] = state_layer
                next_state_layer = self.flip(next_state[i])
                next_state[i] = next_state_layer
                
        action_type = self.get_type_action(action)
        k = random.choice([0, 1, 2, 3])
        
        for i in range(state.shape[0]):
            state_layer = self.rotate(state[i], k=k)
            state[i] = state_layer
            next_state_layer = self.rotate(next_state[i], k=k)
            next_state[i] = next_state_layer
            
        if action_type[0] == 'Move' or action_type[0] == 'Build' or action_type[0] == 'Destroy':
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
                    
                action = self.action_map[(action_type[0], direction)]
                    
            action = self.action_map[(action_type[0], direction)]
    
        return state, action, next_state
    
    def get_valid_actions(self):
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
        
        action_type = self.get_type_action(action)
        current_player = self.state.current_player
        agent_current_idx = self.state.agent_current_idx
        agent_coords_in_order = self.state.agent_coords_in_order
        current_position = agent_coords_in_order[current_player][agent_current_idx]
        previous_scores = self.state.scores
        is_valid_action = self.is_valid_action(action)
        
        if action_type[0] == 'Move':
            direction = action_type[1]
            next_position = (self.direction_map[direction][0] + current_position[0],
                          self.direction_map[direction][1] + current_position[1])
            
            if is_valid_action:
                self.state.agents[current_player][next_position[0]][next_position[1]] = 1
                self.state.agents[current_player][current_position[0]][current_position[1]] = 0
                if self.show_screen:
                    self.screen.draw_agent(next_position[0], next_position[1], current_player)
                    self.screen.make_empty_square(current_position)
            
        elif action_type[0] == 'Build':
            direction = action_type[1]
            wall_coord = (self.direction_map[direction][0] + current_position[0],
                          self.direction_map[direction][1] + current_position[1])
                
            if is_valid_action:
                self.state.walls[current_player][wall_coord[0]][wall_coord[1]] = 1
                if self.show_screen:
                    self.screen.draw_wall(current_player, wall_coord[0], wall_coord[1])
            
        elif action_type[0] == 'Destroy':
            direction = action_type[1]
            wall_coord = (self.direction_map[direction][0] + current_position[0],
                          self.direction_map[direction][1] + current_position[1])
            
            if is_valid_action:
                self.state.walls[0][wall_coord[0]][wall_coord[1]] = 0
                self.state.walls[1][wall_coord[0]][wall_coord[1]] = 0
                if self.show_screen:
                    self.screen.make_empty_square(wall_coord)
        else:
            pass
        
        if self.show_screen:
            self.screen.show_score()
            self.screen.get_numpy_img()
            self.screen.render()
        
        if is_valid_action:
            self.state.update_score()
            new_scores = self.state.scores
            diff_previous_scores = previous_scores[current_player] - previous_scores[1 - current_player]
            diff_new_score = new_scores[current_player] - new_scores[1 - current_player]
            reward = diff_new_score - diff_previous_scores
        else:
            reward = 0
        
        if verbose:
            logging.info('Player: {} | AgentID: {} | Action: {} | Reward: {}'.format(
                current_player, agent_current_idx, action_type, reward))
            
        self.state.agent_current_idx = (agent_current_idx + 1) % self.num_agents
        if self.state.agent_current_idx == 0:
            self.state.current_player = (self.state.current_player + 1) % self.num_players
            self.state.update_agent_coords_in_order()
            if self.state.current_player == 0:
                self.state.remaining_turns -= 1
                
        return self.get_state(), reward, self.game_ended()