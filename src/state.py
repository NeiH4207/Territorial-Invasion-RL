
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
        return {
            'player-id': self.current_player,
            'observation': obs, 
            'curr_agent_xy': current_agent_coords[current_agent_idx]
            }

    def recalculate_score_of_current_player(self):
        """
        Recalculates the score of the current player based on current state
        """
        dx = [0, 0, 1, -1]
        dy = [1, -1, 0, 0]
        
        height = self.height
        width = self.width
        avail = np.zeros((height, width), dtype=int)
        current_player = self.current_player
        st = []
        
        def validate(x, y):
            # if the current land is out of table bounds
            if x >= height or x < 0 or y >= width or y < 0:
                return False
            # if the current land is visited
            if avail[x][y]:
                return False
            # if the current land is a wall
            if self.walls[self.current_player][x][y]:
                return False
            return True
        
        for i in range(height):
            if validate(i, 0):
                st.append((i, 0))
                avail[i][0] = 1
            if validate(i, width - 1):
                st.append((i, width - 1))
                avail[i][width - 1] = 1
        
        for j in range(width):
            if validate(0, j):
                st.append((0, j))
                avail[0][j] = 1
            if validate(height - 1, j):
                st.append((height - 1, j))
                avail[height - 1][j] = 1
            
        while len(st) > 0:
            x, y = st.pop()
            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]
                if validate(nx, ny):
                    avail[nx][ny] = 1
                    st.append((nx, ny))
        
        return height * width - avail.sum() - self.walls[current_player].sum()