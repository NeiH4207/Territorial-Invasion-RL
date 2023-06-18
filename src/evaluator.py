from itertools import count
import logging
import numpy as np

import torch
from tqdm import tqdm

from src.environment import AgentFighting


class Evaluator():
    def __init__(self, env: AgentFighting, n_evals=10, device=None):
        self.env = env
        self.n_evals = n_evals
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 0 ~ 999: K = 30; 1000 ~ 1999: K = 15; 2000 ~ 2999: K = 10; 3000 ~ : K = 5
        self.K_TABLE = [30, 15, 10, 5]   

        self.R_PRI = 40

    def compute_elo(self, r0, r1, w):
        '''
        Compute the elo rating with method from http://www.xqbase.com/protocol/elostat.htm
        r0: red player's elo rating
        r1: black player's elo rating
        w: game result: 1 = red win, 0.5 = draw, 0 = black win
        '''
        relative_elo = r1 - r0 - self.R_PRI
        we = 1 / (1 + 10 ** (relative_elo / 400))
        k0 = self.K_TABLE[-1] if r0 >= 3000 else self.K_TABLE[r0 // 1000]
        k1 = self.K_TABLE[-1] if r1 >= 3000 else self.K_TABLE[r1 // 1000]
        rn0 = int(r0 + k0 * (w - we))
        rn1 = int(r1 + k1 * (we - w))
        rn0 = rn0 if rn0 > 0 else 0
        rn1 = rn1 if rn1 > 0 else 0
        return (rn0, rn1)
        
    def eval(self, old_model, new_model, change_elo=True):
        
        elo_1 = old_model.get_elo()
        elo_2 = new_model.get_elo()
        old_elo = elo_2
        num_wins = 0
        
        _tqdm = tqdm(range(self.n_evals), desc='Evaluating (Win 0/{})'.format(self.n_evals))
        for i in _tqdm:
            done = False
            state = self.env.get_state()
            for cnt in count():
                if state['player-id'] == 0:
                    valid_actions = self.env.get_valid_actions()
                    torch_state = torch.FloatTensor(state['observation']).to(self.device)
                    act_values = new_model.predict(torch_state)[0]
                    if valid_actions is not None:
                        act_values[~valid_actions] = -float('inf')
                    action = int(np.argmax(act_values))
                else:
                    valid_actions = self.env.get_valid_actions()
                    torch_state = torch.FloatTensor(state['observation']).to(self.device)
                    act_values = old_model.predict(torch_state)[0]
                    if valid_actions is not None:
                        act_values[~valid_actions] = -float('inf')
                    action = int(np.argmax(act_values))
                scores = self.env.state.scores
                _tqdm.set_postfix_str(f'Scores: {scores[0]} / {scores[1]}')
                next_state, _, done = self.env.step(action)
                state = next_state
                if done:
                    break
            winner = self.env.get_winner()
            
            if winner == 1:
                num_wins += 1
                
            if winner == 1:
                score = 1
            elif winner == -1:
                score = 0
            else:
                score = 0.5
                
            _tqdm.set_description(f'Evaluating (Win {num_wins}/{self.n_evals})')
            elo_1, elo_2 = self.compute_elo(elo_1, elo_2, score)
            if i < self.n_evals - 1:
                self.env.reset()
        
        if change_elo:
            old_model.set_elo(elo_1)
            new_model.set_elo(elo_2)
            logging.info('Elo changes from {} to {} | Win {}/{}'.\
                format(old_elo, elo_2, num_wins, self.n_evals))
        else:
            won_player = 1 if num_wins <= self.n_evals - num_wins else 2
            num_wins = num_wins if won_player == 2 else self.n_evals - num_wins
        return num_wins / self.n_evals >= 0.55