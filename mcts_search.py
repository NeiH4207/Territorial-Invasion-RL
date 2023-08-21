"""
Created on Tue Apr 27 2:19:47 2023
@author: hien
"""
from __future__ import division
from copy import deepcopy
from itertools import count
import json
import logging
import os
import time
import torch
from Algorithms.Rainbow import Rainbow
from src.mcts import MCTS
from src.evaluator import Evaluator
from src.environment import AgentFighting
from src.utils import *
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
from argparse import ArgumentParser
from models.AgentDQN import DQN
from Algorithms.DDQN import DDQN
from Algorithms.PER import PER

def argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--show-screen', type=bool, default=True)
    parser.add_argument('-a', '--algorithm', default='mcts')
    parser.add_argument('-v', '--verbose', action='store_true', default=True)
    parser.add_argument('--figure-path', type=str, default='figures/')
    parser.add_argument('--n-evals', type=int, default=50)
    
    # DDQN arguments
    parser.add_argument('--gamma', type=float, default=0.975)
    parser.add_argument('--tau', type=int, default=0.01)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--exploit-rate', type=float, default=0.25)
    
    # model training arguments
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--memory-size', type=int, default=32768 * 4)
    parser.add_argument('--num-episodes', type=int, default=100000)
    parser.add_argument('--model-path', type=str, default='trained_models/procon.pt')
    
    return parser.parse_args()

def main():
    args = argument_parser()
    configs = json.load(open('configs/map.json'))
    env = AgentFighting(args, configs, args.show_screen)
    observation_shape = env.get_space_size()
    n_actions = env.n_actions
    logging.info('Observation space: {}'.format(observation_shape))
    logging.info('Action space: {}'.format(n_actions))
    algorithm = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DQN(observation_shape, n_actions, 
                optimizer=args.optimizer, 
                lr=args.lr,
                dueling=True).to(device)
    
    model.load(args.model_path, device)
    
    algorithm = MCTS(env, model,
            numMCTSSims=50,
            cpuct=1,
            exploration_rate=0,
            selfplay=True)
    
    state = env.get_state(obj=True)
    env.render(state)
    while state.is_terminal() == False:
        start = time.time()
        action = algorithm.get_action(state)
        ed = time.time()
        logging.info('Action: {} | Time: {}'.format(action, ed - start))
        state.next(action)
        env.render(state)

if __name__ == "__main__":
    main()