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
from Algorithms.RandomStep import RandomStep
from models.AZNet import AZNet
from src.environment import AgentFighting
log = logging.getLogger(__name__)
from argparse import ArgumentParser
from Algorithms.DQN import DQN
import matplotlib
is_ipython = 'inline' in matplotlib.get_backend()

plt.ion()

def argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--show-screen', type=bool, default=True)
    parser.add_argument('-a', '--algorithm', default='dqn')
    parser.add_argument('-n', '--num-game', default=1000, type=int)
    parser.add_argument('-v', '--verbose', action='store_true', default=True)
    parser.add_argument('--model-path', type=str, default='trained_models/nnet.pt')
    parser.add_argument('--load-model', action='store_true', default=True)
    return parser.parse_args()

def main():
    args = argument_parser()
    configs = json.load(open('config.json'))
    env = AgentFighting(args, configs, args.show_screen)
    n_observations = env.get_space_size()
    n_actions = env.n_actions
    model = AZNet(n_observations, n_actions)
    dqn = DQN(n_observations, 
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
        dqn.load_model(args.model_path)
        
    random_step = RandomStep(n_actions=env.n_actions, num_agents=env.num_agents)
    
    for episode in range(args.num_game):
        done = False
        state = env.reset()
        state = state
        for cnt in count():
            env.render()
            if state['player-id'] == 0:
                action = random_step.get_action(state['observation'])
            else:
                action = dqn.get_action(state['observation'])
            next_state, reward, done = env.step(action)
            state = next_state
            if done:
                break
            time.sleep(0.1)
            
        dqn.save_model(args.model_path)
        print('Episode {} finished after {} timesteps.'.format(episode, cnt))
                
    time.sleep(3)

if __name__ == "__main__":
    main()