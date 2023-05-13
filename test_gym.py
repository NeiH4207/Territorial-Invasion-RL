"""
Created on Tue Apr 27 2:19:47 2023
@author: hien
"""
from __future__ import division
from itertools import count
import json
import logging
import os
import random
import time

import numpy as np
from Algorithms.RandomStep import RandomStep
log = logging.getLogger(__name__)
from argparse import ArgumentParser
from Algorithms.DQN import DQN
import gym

def argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--show-screen', type=bool, default=True)
    parser.add_argument('-a', '--algorithm', default='dqn')
    parser.add_argument('-n', '--num-game', default=1000, type=int)
    parser.add_argument('-v', '--verbose', action='store_true', default=True)
    parser.add_argument('--model-path', type=str, default='trained_models/nnet.pt')
    parser.add_argument('--load-model', action='store_true', default=False)
    return parser.parse_args()

def main():
    args = argument_parser()
    configs = json.load(open('config.json'))
    env = gym.make('CartPole-v1', render_mode='human')
    n_observations, n_actions = env.observation_space.shape[0], env.action_space.n
    algorithm = None
    if args.algorithm == 'dqn':
        algorithm = DQN(n_observations, 
                        n_actions,
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
        state = env.reset()
        state = np.reshape(state[0], [1,4])
        cnt = 0
        for cnt in count():
            env.render()
            action = algorithm.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            reward = reward if not done else -1
            next_state = np.reshape(next_state, [1,4])
            algorithm.memorize(state, action, next_state, reward, done)
            state = next_state
            algorithm.replay(configs['model']['batch_size'])
            if done or truncated:
                break
            
        print('Episode {} finished after {} timesteps.'.format(episode, cnt))
                
    time.sleep(3)

if __name__ == "__main__":
    main()