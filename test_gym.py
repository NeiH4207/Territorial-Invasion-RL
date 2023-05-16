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

import numpy as np
import torch
from Algorithms.RandomStep import RandomStep
log = logging.getLogger(__name__)
from argparse import ArgumentParser
from Algorithms.DQN import DQN
from Algorithms.DDQN import DDQN
from models.GymNet import GymNet
import gym

def argument_parser():
    parser = ArgumentParser()
    parser.add_argument('-a', '--algorithm', default='dqn')
    parser.add_argument('-n', '--num-game', default=1000, type=int)
    parser.add_argument('--model-path', type=str, default='trained_models/nnet.pt')
    parser.add_argument('--load-model', action='store_true', default=False)
    return parser.parse_args()

def main():
    args = argument_parser()
    configs = json.load(open('config.json'))
    env = gym.make('CartPole-v1')
    n_observations, n_actions = env.observation_space.shape[0], env.action_space.n
    algorithm = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GymNet(n_observations, n_actions).to(device)
    if args.algorithm == 'dqn':
        algorithm = DDQN(n_observations, 
                        n_actions,
                        model,
                        configs['model']['optimizer'],
                        configs['model']['lr'],
                        model_path=args.model_path)
        model_dir = os.path.dirname(args.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logging.info('Created model directory: {}'.format(model_dir))
        if args.load_model:
            algorithm.load_model(args.model_path)
            
    elif args.algorithm == 'random':
        algorithm = RandomStep(n_actions=env.n_actions, num_agents=env.num_agents)
    else:
        raise ValueError('Algorithm {} is not supported'.format(args.algorithm))
    
    for episode in range(args.num_game):
        done = False
        state, info = env.reset()
        cnt = 0
        for cnt in count():
            env.render()
            action = algorithm.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            reward = reward if not done else -1
            algorithm.memorize(state, action, next_state, reward, done)
            state = next_state
            if done or truncated:
                break
            
        algorithm.adaptiveEGreedy()
        algorithm.replay(configs['model']['batch_size'])
        print('Episode {} finished after {} timesteps.'.format(episode, cnt))
                
    time.sleep(3)

if __name__ == "__main__":
    main()