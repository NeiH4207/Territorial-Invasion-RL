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
from models.DQN import DQN
from src.environment import AgentFighting
from src.utils import plot_history
log = logging.getLogger(__name__)
from argparse import ArgumentParser
from Algorithms.DDQN import DDQN

def argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--show-screen', type=bool, default=False)
    parser.add_argument('-a', '--algorithm', default='dqn')
    parser.add_argument('-v', '--verbose', action='store_true', default=True)
    parser.add_argument('--figure-path', type=str, default='figures/')
    
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
    parser.add_argument('--num-episodes', type=int, default=100000)
    parser.add_argument('--model-path', type=str, default='trained_models/nnet.pt')
    parser.add_argument('--load-model', action='store_true', default=False)
    
    return parser.parse_args()

def main():
    args = argument_parser()
    configs = json.load(open('config.json'))
    env = AgentFighting(args, configs, args.show_screen)
    n_observations = env.get_space_size()
    n_actions = env.n_actions
    algorithm = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DQN(n_observations, n_actions, dueling=True).to(device)
    
    model_dir = os.path.dirname(args.model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logging.info('Created model directory: {}'.format(model_dir))

        
    if not os.path.exists(args.figure_path):
        os.makedirs(args.figure_path)
        logging.info('Created figure directory: {}'.format(args.figure_path))
        
    if args.algorithm == 'dqn':
        algorithm = DDQN(   n_observations=n_observations, 
                            n_actions=n_actions,
                            model=model,
                            optimizer=args.optimizer,
                            lr=args.lr,
                            tau=args.tau,
                            gamma=args.gamma,
                            epsilon=args.epsilon,
                            epsilon_min=args.epsilon_min,
                            epsilon_decay=args.epsilon_decay,
                            memory_size=args.memory_size,
                            model_path=args.model_path
                        )
        if args.load_model:
            algorithm.load_model(args.model_path)
            
    elif args.algorithm == 'pso':
        return
    
    elif args.algorithm == 'random':
        algorithm = RandomStep(n_actions=env.n_actions, num_agents=env.num_agents)
        
    else:
        raise ValueError('Algorithm {} is not supported'.format(args.algorithm))
    
    for episode in range(args.num_episodes):
        done = False
        state = env.get_state()
        state = state['observation']
        for cnt in count():
            env.render()
            valid_actions = env.get_valid_actions()
            action = algorithm.get_action(state, valid_actions)
            next_state, reward, done = env.step(action)
            next_state = next_state['observation']
            algorithm.memorize(state, action, next_state, reward, done)
            state = next_state
            env.save_image(os.path.join(args.figure_path, 'current_state.png'.format(episode, cnt)))
            if done:
                print(env.state.scores)
                break
        history = algorithm.replay(args.batch_size, verbose=args.verbose)
        if history and args.verbose:
            plot_history(history, args.figure_path)
        env.reset()
        algorithm.adaptiveEGreedy()
        algorithm.save_model(args.model_path)
        print('Episode {} finished after {} timesteps.'.format(episode, cnt))

if __name__ == "__main__":
    main()