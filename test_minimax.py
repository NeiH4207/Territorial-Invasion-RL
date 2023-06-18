"""
Created on Tue Apr 27 2:19:47 2023
@author: hien
"""
from __future__ import division
import json
import logging
import time

import torch
from Algorithms.RandomStep import RandomStep
from Algorithms.Minimax import Minimax
from models.AgentDQN import DQN
from src.environment import AgentFighting
log = logging.getLogger(__name__)
from random import seed
from argparse import ArgumentParser

def argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--show-screen', type=bool, default=True)
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--model-path', type=str, default='trained_models/nnet2.pt')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def main():
    args = argument_parser()
    configs = json.load(open('configs/map.json'))
    env = AgentFighting(args, configs, args.show_screen)
    n_observations = env.get_space_size()
    n_actions = env.n_actions
    
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    model = DQN(n_observations, n_actions).to(device)
    model.load(args.model_path)
    
    algorithm = Minimax(env, model, max_depth=7)
    env.reset()
    if args.show_screen:
        env.render()
    state = env.get_state(obj=True)
    while not env.is_terminal():
        action = algorithm.get_action(state)
        _ = env.step(action, verbose=True)
        env.render()
        state = env.get_state(obj=True)
    
    winner = env.get_winner()
    if winner == -1:
        logging.info('Game ended. Draw')
    else:
        logging.info('Game ended. Winner: {}'.format(env.get_winner()))

if __name__ == "__main__":
    main()