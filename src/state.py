
import numpy as np
from src.map import Map
from copy import deepcopy as dcopy

class State(Map):
    def __init__(self, configs):
        super().__init__(configs)
        self.num_players = 2
        self.current_player = 0
        self.num_agents = None
    
    def set_players(self, players):
        self.players = players
    
    def get_agent_position(self):
        return self.agent_pos[self.current_player]
    
    def string_resentation(self):
        """
        Returns a hash code for string representation of the state
        """
        s = str(self.agent_pos) + \
            str(self.castles_remaining) + \
            str(self.territory_board)
        return hash(s)
    
    def to_opponent(self):
        state = dcopy(self)
        state.current_player ^= 1
        return state
    def get_state(self):
        # Standardized variable names to improve readability
        players = [self.current_player, self.current_player ^ 1]
        agent_board = self.agents[players]
        castle_board = self.castles
        wall_board = self.walls[players]
        territory_board = self.territories[players]

        # Removed debugging statement
        obs = np.stack(
            (
                agent_board[0], 
                agent_board[1],
                castle_board,
                wall_board[0], 
                wall_board[1],
                territory_board[0], 
                territory_board[1]
            ),
            axis=0
        )

        # Standardized variable names to improve readability and changed key name
        current_agent_coords = self.agent_coords_in_order[self.current_player]
        current_agent_idx = self.agent_current_idx
        return {'obs': obs, 'coord_agent_in_action': current_agent_coords[current_agent_idx]}
