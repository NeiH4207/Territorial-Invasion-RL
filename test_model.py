"""
Created on Tue Apr 27 2:19:47 2023
@author: hien
"""
from __future__ import division
from itertools import count
import json
import logging
import os
import time
from matplotlib import pyplot as plt
import torch
from Algorithms.RandomStep import RandomStep
from src.evaluator import Evaluator
from models.AgentDQN import DQN
from src.environment import AgentFighting
log = logging.getLogger(__name__)
from argparse import ArgumentParser
from Algorithms.DDQN import DDQN
import matplotlib
is_ipython = 'inline' in matplotlib.get_backend()

plt.ion()

def argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--show-screen', type=bool, default=True)
    parser.add_argument('-a', '--algorithm', default='dqn')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--figure-path', type=str, default='figures/')
    parser.add_argument('--n-evals', type=int, default=5)
    
    # DDQN arguments
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=int, default=0.005)
    parser.add_argument('--epsilon', type=float, default=0.9)
    parser.add_argument('--epsilon-min', type=float, default=0.1)
    parser.add_argument('--epsilon-decay', type=float, default=0.995)
    
    # model training arguments
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--memory-size', type=int, default=32768)
    parser.add_argument('--num-episodes', type=int, default=10)
    parser.add_argument('--model-path-1', type=str, default='trained_models/nnet.pt')
    parser.add_argument('--model-path-2', type=str, default='trained_models/nnet.pt')
    parser.add_argument('--load-model', action='store_true', default=True)
    return parser.parse_args()

def main():
    args = argument_parser()
    configs = json.load(open('config.json'))
    env = AgentFighting(args, configs, args.show_screen)
    n_observations = env.get_space_size()
    n_actions = env.n_actions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_1 = DQN(n_observations, n_actions, dueling=True).to(device)
    model_2 = DQN(n_observations, n_actions, dueling=True).to(device)
    
    if args.load_model:
        model_1.load(args.model_path_1)
        model_2.load(args.model_path_2)
    
    evaluator = Evaluator(env, n_evals=args.n_evals, device=device)
    
    evaluator.eval(model_1, model_2, change_elo=False)

if __name__ == "__main__":
    main()