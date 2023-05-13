from copy import deepcopy as dcopy
import logging
from GameBoard.screen import Screen
from src.player import Player
from src.algorithms import *
from src.state import State
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.NOTSET)

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
        if self.args.show_screen:
            self.screen = Screen(self)
        self.players = [Player(i, self.num_players) for i in range(self.num_players)]
        self.current_player = 0
        self.reset()
        self.num_agents = self.state.num_agents
        
    def render(self):
        if self.show_screen:
            self.screen.render()
    
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
        if self.show_screen:
            self.screen.init(self.state)
        self.num_agents = self.state.num_agents
        return self.state.get_state()
    
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
        return self.state.get_state()['observation'].shape
            
    def get_state(self):
        return dcopy(self.state.get_state())
    
    def get_score(self):
        """
        Returns the sum of walls in each player's side of the board.

        :return: a list containing the sum of walls in each player's side of the board.
        """
        return [self.state.walls[0].sum(), self.state.walls[1].sum()]
    
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
        scores = self.get_score()
        if scores[0] > scores[1]:
            return 0
        elif scores[1] > scores[0]:
            return 1
        else:
            return -1
        
    def step(self, action):
        """
        This function performs a single step of the game by taking an action as input. The action 
        should be valid or else the function returns the reward. If the action is valid, then the 
        function updates the state of the game and returns the reward.

        Args:
            action: The action to be taken in the game.

        Returns:
            reward: The reward obtained from the step.
        """
        if not self.is_valid_action(action):
            logging.warning('Invalid action! - ' + str(action))
            return self.get_state(), 0, self.game_ended()
        action_type = self.get_type_action(action)
        current_player = self.state.current_player
        agent_current_idx = self.state.agent_current_idx
        agent_coords_in_order = self.state.agent_coords_in_order
        current_coord = agent_coords_in_order[current_player][agent_current_idx]
        is_valid_action = True
        old_scores = self.get_score()
        old_diff_score = old_scores[current_player] - old_scores[1 - current_player]
        
        if action_type[0] == 'Move':
            direction = action_type[1]
            next_coord = (self.direction_map[direction][0] + current_coord[0],
                          self.direction_map[direction][1] + current_coord[1])
            if not self.in_bounds(next_coord):
                is_valid_action = False
                
            elif next_coord in self.state.agent_coords_in_order[0] or \
                        next_coord in self.state.agent_coords_in_order[1]:
                is_valid_action = False
                
            elif self.state.agents[current_player][next_coord[0]][next_coord[1]] == 1:
                ''' in turn (N agent actions at the same time), only one agent can move at an area, 
                    so the other agent is moved into the same area befores
                    agents save next coordinates but agent_coords_in_order is not updated to check this '''
                is_valid_action = False
                
            elif self.state.walls[0][next_coord[0]][next_coord[1]] == 1 \
                    or self.state.walls[1][next_coord[0]][next_coord[1]] == 1:
                is_valid_action = False
                
            elif self.state.castles[next_coord[0]][next_coord[1]] == 1 \
                    or self.state.castles[next_coord[0]][next_coord[1]] == 1:
                is_valid_action = False
                
            if is_valid_action:
                self.state.agents[current_player][next_coord[0]][next_coord[1]] = 1
                self.state.agents[current_player][current_coord[0]][current_coord[1]] = 0
                if self.show_screen:
                    self.screen.draw_agent(next_coord[0], next_coord[1], current_player)
                    self.screen.make_empty_square(current_coord)
            
        elif action_type[0] == 'Build':
            direction = action_type[1]
            wall_coord = (self.direction_map[direction][0] + current_coord[0],
                          self.direction_map[direction][1] + current_coord[1])
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
                
            if is_valid_action:
                self.state.walls[current_player][wall_coord[0]][wall_coord[1]] = 1
                if self.show_screen:
                    self.screen.draw_wall(current_player, wall_coord[0], wall_coord[1])
            
        elif action_type[0] == 'Destroy':
            direction = action_type[1]
            wall_coord = (self.direction_map[direction][0] + current_coord[0],
                          self.direction_map[direction][1] + current_coord[1])
            if not self.in_bounds(wall_coord):
                is_valid_action = False
                
            elif self.state.walls[0][wall_coord[0]][wall_coord[1]] == 0 \
                    and self.state.walls[1][wall_coord[0]][wall_coord[1]] == 0:
                is_valid_action = False
                
            if is_valid_action:
                self.state.walls[0][wall_coord[0]][wall_coord[1]] = 0
                self.state.walls[1][wall_coord[0]][wall_coord[1]] = 0
                if self.show_screen:
                    self.screen.make_empty_square(wall_coord)
        else:
            pass
        
        new_scores = self.get_score()
        new_diff_score = new_scores[current_player] - new_scores[1 - current_player]
        reward = new_diff_score - old_diff_score
        self.state.players[0].score = new_scores[0]
        self.state.players[1].score = new_scores[1]
        
        self.state.agent_current_idx = (agent_current_idx + 1) % self.num_agents
        if self.state.agent_current_idx == 0:
            self.state.current_player = (self.state.current_player + 1) % self.num_players
            self.state.update_agent_coords_in_order()
            if self.state.current_player == 0:
                self.state.remaining_turns -= 1
                
        if self.show_screen:
            self.screen.show_score()
            self.screen.render()
            
        if is_valid_action:
            logging.info('Player: {} | AgentID: {} | Action: {} | Reward: {}'.format(
                self.state.current_player, self.state.agent_current_idx, action_type, reward))
            return self.get_state(), 0, self.game_ended()
        return self.get_state(), reward, self.game_ended()