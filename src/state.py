
import numpy as np
from src.map import Map
from copy import deepcopy as dcopy

class State(Map):
    def __init__(self, configs):
        super().__init__(configs)
        self.num_players = 2
        self.current_player = 0
        self.num_agents = None
        self.wall_scores = [0 for _ in range(self.num_players)]
        self.castle_scores = [0 for _ in range(self.num_players)]
        self.open_territory_scores = [0 for _ in range(self.num_players)]
        self.closed_territory_scores = [0 for _ in range(self.num_players)]
        self.territory_scores = [0 for _ in range(self.num_players)]
        self.alpha = 1
        self.beta = 10
        self.gamma = 1
        
    @property
    def scores(self):
        score_A = self.alpha * self.wall_scores[0] + self.beta * self.castle_scores[0] + \
            (self.open_territory_scores[0] + self.closed_territory_scores[0])
        score_B = self.alpha * self.wall_scores[1] + self.beta * self.castle_scores[1] + \
            (self.open_territory_scores[1] + self.closed_territory_scores[1])
        return np.array([score_A, score_B])
    
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
        agent_current_board = np.zeros(agent_board[0].shape)
        # Removed debugging statement
        obs = np.stack(
            (
                agent_board[0], 
                agent_board[1],
                castle_board,
                wall_board[0], 
                wall_board[1],
                territory_board[0], 
                territory_board[1],
                agent_current_board
            ),
            axis=0
        )

        # Standardized variable names to improve readability and changed key name
        current_agent_idx = self.agent_current_idx
        current_agent_coords = self.agent_coords_in_order[self.current_player][current_agent_idx]
        agent_current_board[current_agent_coords[0], current_agent_coords[1]] = 1
        return {
            'player-id': self.current_player,
            'observation': obs, 
            'curr_agent_xy': current_agent_coords[current_agent_idx]
            }

    def get_scores(self, player):
        """
        Recalculates the score of the current player based on current state
        """
        dx = [0, 0, 1, -1]
        dy = [1, -1, 0, 0]
        
        height = self.height
        width = self.width
        avail = np.zeros((height, width), dtype=int)
        st = []
        opponent = 1 - player
        
        def validate(x, y):
            # if the current land is out of table bounds
            if x >= height or x < 0 or y >= width or y < 0:
                return False
            # if the current land is visited
            if avail[x][y]:
                return False
            # if the current land is a wall
            if self.walls[player][x][y]:
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
        
        wall_score = self.walls[player].sum()
        
        closed_territory_score = height * width - avail.sum() - wall_score

        castle_score = 0
        
        for i in range(height):
            for j in range(width):
                if self.walls[opponent][i][j] == 1 and self.territories[player][i][j] == 1:
                    self.territories[player][i][j] = 0
                if avail[i][j] == 0 and self.walls[player][i][j] == 0:
                    self.territories[player][i][j] = 1
                    if self.castles[i][j] == 1:
                        castle_score += 1
        
        all_territory_score = self.territories[player].sum()
        
        open_territory_score = all_territory_score - closed_territory_score
        

        return wall_score, closed_territory_score, open_territory_score, castle_score
    
    def update_score(self):
        """
        Updates the score of the current player based on current state
        """
        for player in range(self.num_players):
            wall_score, closed_territory_score, open_territory_score, castle_score = self.get_scores(player)
            self.wall_scores[player] = wall_score
            self.closed_territory_scores[player] = closed_territory_score
            self.open_territory_scores[player] = open_territory_score
            self.territory_scores[player] = open_territory_score + closed_territory_score
            self.castle_scores[player] = castle_score
            
        for player in range(self.num_players):
            self.players[player].scores = self.scores[player]