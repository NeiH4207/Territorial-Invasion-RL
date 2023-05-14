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
    parser.add_argument('--figure-path', type=str, default='figures/')
    parser.add_argument('--load-model', action='store_true', default=True)
    return parser.parse_args()

def main():
    args = argument_parser()
    configs = json.load(open('config.json'))
    env = AgentFighting(args, configs, args.show_screen)
    n_observations = env.get_space_size()
    n_actions = env.n_actions
    algorithm = None
    model = AZNet(n_observations, n_actions)
    
    model_dir = os.path.dirname(args.model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logging.info('Created model directory: {}'.format(model_dir))

        
    if not os.path.exists(args.figure_path):
        os.makedirs(args.figure_path)
        logging.info('Created figure directory: {}'.format(args.figure_path))
        
    if args.algorithm == 'dqn':
        algorithm = DQN(n_observations, 
                        n_actions,
                        model,
                        configs['model']['optimizer'],
                        configs['model']['lr'],
                        model_path=args.model_path)
        if args.load_model:
            algorithm.load_model(args.model_path)
            
    elif args.algorithm == 'random':
        algorithm = RandomStep(n_actions=env.n_actions, num_agents=env.num_agents)
    else:
        raise ValueError('Algorithm {} is not supported'.format(args.algorithm))
    
    for episode in range(args.num_game):
        done = False
        state = env.reset()
        state = state['observation']
        for cnt in count():
            env.render()
            action = algorithm.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = next_state['observation']
            algorithm.memorize(state, action, next_state, reward, done)
            state = next_state
            history = algorithm.replay(configs['model']['batch_size'])
            if history and args.verbose:
                plt.figure(1)
                plt.xlabel('Episode')
                plt.ylabel('Loss')
                plt.title('Training...')
                plt.ion()
                plt.plot(history['loss'], 'b')
                plt.savefig(os.path.join(args.figure_path, 'loss.png'))
            if done:
                break
            
        algorithm.save_model(args.model_path)
        print('Episode {} finished after {} timesteps.'.format(episode, cnt))
                
    time.sleep(3)

if __name__ == "__main__":
    main()