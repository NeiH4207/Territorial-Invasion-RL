
import numpy as np
import torch

from src.state import State


class Minimax():
    def __init__(self, model=None, handicap=4, max_depth=4):
        self.model = model
        self.handicap = handicap
        self.max_depth = max_depth
        
    def get_all_valid_actions(self, state: State):
        valid_actions = []
        for action in range(state.n_actions):
            if state.is_valid_action(action):
                valid_actions.append(action)
        return valid_actions
        
    def get_top_moves(self, state: State, n: int):
        obs = state.get_state()['observation']
        evaluations = self.model.predict(obs)
        n_valids = 0
        for action in range(state.n_actions):
            if not state.is_valid_action(action):
                evaluations[action] = -9999
            else:
                n_valids += 1
        return evaluations.argsort()[::-1][:min(n, n_valids)].tolist()
    
    def get_action(self, state: State):
        best_value = -9999

        top_actions = self.get_top_moves(state, self.handicap * 2)
        best_action = top_actions[0]
        
        for action in top_actions:
            next_state = state.copy()
            next_state.next(action)
            _is_max_state = True
            if state.agent_current_idx == state.num_agents - 1:
                _is_max_state = False
            value = self.minimax(next_state, -10e5, 10e5, self.max_depth, state.current_player, _is_max_state)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action
    
    def minimax(self, state, alpha, beta, depth, player, is_max_state):
        if depth == 0 or state.is_terminal():
            if depth == 0:
                obs = state.get_state()['observation']
                return self.model.predict(obs).max()
            return state.scores[player] - state.scores[player ^ 1]
        
        if is_max_state:
            value = -9999
            for action in self.get_top_moves(state, self.handicap):
                next_state = state.copy()
                next_state.next(action)
                _is_max_state = is_max_state
                if state.agent_current_idx == state.num_agents - 1:
                    _is_max_state = not is_max_state
                value = max(
                    value,
                    self.minimax(next_state, alpha, beta, depth - 1, player, _is_max_state)
                )
                alpha = max(value, alpha)
                if alpha >= beta:
                    break
            return value
        else:
            value = 9999
            for action in self.get_top_moves(state, self.handicap):
                next_state = state.copy()
                next_state.next(action)
                _is_max_state = is_max_state
                if state.agent_current_idx == state.num_agents - 1:
                    _is_max_state = not is_max_state
                value = min(
                    value,
                    self.minimax(next_state, alpha, beta, depth - 1, player, _is_max_state)
                )
                beta = min(value, beta)
                if alpha >= beta:
                    break
            return value
