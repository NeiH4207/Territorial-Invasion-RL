
import numpy as np
import torch

from src.state import State


class Minimax():
    def __init__(self, env, model, handicap=4, max_depth=4):
        self.env = env
        self.model = model
        self.handicap = handicap
        self.max_depth = max_depth
        
    def get_all_valid_actions(self, state: State):
        valid_actions = []
        for action in range(self.env.n_actions):
            if state.is_valid_action(action):
                valid_actions.append(action)
        return valid_actions
        
    def get_top_moves(self, state: State, n: int, is_max_state: bool):
        obs = state.get_state()['observation']
        obs = torch.from_numpy(obs).float().to(self.model.get_device())
        evaluations = self.model(obs.unsqueeze(0))[0]
        for action in range(self.env.n_actions):
            if not state.is_valid_action(action):
                evaluations[action] = -9999 if is_max_state else 9999
        return evaluations.argsort(descending=is_max_state)[:n].tolist()
    
    def get_action(self, state: State):
        best_value = -9999

        top_actions = self.get_top_moves(state, self.handicap, True)
        
        for action in top_actions:
            next_state = state.copy()
            next_state.next(action)
            is_next_player = next_state.agent_current_idx == 0
            is_max_state = False if is_next_player else True
            value = self.minimax(next_state, -10e5, 10e5, self.max_depth - 1, is_max_state)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action
    
    def minimax(self, state, alpha, beta, depth, is_max_state):
        if depth == 0 or state.is_terminal():
            curr_player = state.get_curr_player()
            diff = state.scores[curr_player] - state.scores[curr_player ^ 1]
            return diff

        if is_max_state:
            value = -9999
            for action in self.get_top_moves(state, self.handicap, is_max_state):
                if state.is_valid_action(action):
                    next_state = state.copy()
                    next_state.next(action)
                    is_next_player = next_state.agent_current_idx == 0
                    _is_max_state = not is_max_state if is_next_player else is_max_state
                    value = max(
                        value,
                       self. minimax(next_state, alpha, beta, depth - 1, _is_max_state)
                    )
                    alpha = max(value, alpha)
                    if alpha >= beta:
                        break
            return value
        else:
            value = 9999
            for action in self.get_top_moves(state, self.handicap, is_max_state):
                if state.is_valid_action(action):
                    next_state = state.copy()
                    next_state.next(action)
                    is_next_player = next_state.agent_current_idx == 0
                    _is_max_state = not is_max_state if is_next_player else is_max_state
                    value = min(
                        value,
                        self.minimax(next_state, alpha, beta, depth - 1, _is_max_state)
                    )
                    beta = min(value, beta)
                    if alpha >= beta:
                        break
            return value
