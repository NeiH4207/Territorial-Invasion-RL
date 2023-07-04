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
import torch
from Algorithms.Rainbow import Rainbow
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
    parser.add_argument('--show-screen', type=bool, default=False)
    parser.add_argument('-a', '--algorithm', default='rainbow')
    parser.add_argument('-v', '--verbose', action='store_true', default=True)
    parser.add_argument('--figure-path', type=str, default='figures/')
    parser.add_argument('--n-evals', type=int, default=50)
    
    # DDQN arguments
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=int, default=0.01)
    parser.add_argument('--n-step', type=int, default=3)
    
    # model training arguments
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--memory-size', type=int, default=32768)
    parser.add_argument('--num-episodes', type=int, default=100000)
    parser.add_argument('--model-path', type=str, default='trained_models/nnet.pt')
    parser.add_argument('--load-model', action='store_true')
    
    return parser.parse_args()

def main():
    args = argument_parser()
    configs = json.load(open('configs/map.json'))
    env = AgentFighting(args, configs, args.show_screen)
    n_observations = env.get_space_size()
    n_actions = env.n_actions
    algorithm = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DQN(n_observations, n_actions, 
                optimizer=args.optimizer, 
                lr=args.lr,
                dueling=True).to(device)
    if args.load_model:
        model.load(args.model_path, device)
    evaluator = Evaluator(AgentFighting(args, configs, False), n_evals=args.n_evals, device=device)
    
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
                            tau=args.tau,
                            gamma=args.gamma,
                            memory_size=args.memory_size,
                            model_path=args.model_path
                        )
    elif args.algorithm == 'per':
        algorithm = PER(   n_observations=n_observations, 
                            n_actions=n_actions,
                            model=model,
                            tau=args.tau,
                            gamma=args.gamma,
                            memory_size=args.memory_size,
                            model_path=args.model_path,
                            batch_size=args.batch_size,
                            alpha=0.2,
                            beta=0.6,
                            prior_eps=1e-6
                        )
    elif args.algorithm == 'rainbow':
        algorithm = Rainbow(n_observations=n_observations, 
                            n_actions=n_actions,
                            model=model,
                            tau=args.tau,
                            gamma=args.gamma,
                            memory_size=args.memory_size,
                            model_path=args.model_path,
                            batch_size=args.batch_size,
                            alpha=0.2,
                            beta=0.6,
                            prior_eps=1e-6,
                            n_step=args.n_step
                        )
        
    else:
        raise ValueError('Algorithm {} is not supported'.format(args.algorithm))
    
    best_model_path = args.model_path.replace('.pt', '_best.pt')
    model.save(best_model_path)
    
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
            # state, action, next_state = env.get_symmetry_transition(state, action, next_state)
            transition = [state, action, reward, next_state, done]
            one_step_transition = algorithm.memory_n.store(*transition)
            if one_step_transition:
                algorithm.memory.store(*one_step_transition)
            algorithm.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            
        if episode % 3 == 0 and algorithm.fully_mem(0.25):
            history_loss = algorithm.replay(args.batch_size, verbose=args.verbose)
            plot_timeseries(history_loss, args.figure_path, 'episode', 'loss', 'Training Loss')
        
        if (episode + 1) % 100 == 0:
            best_model = DQN(n_observations, n_actions, dueling=True).to(device)
            best_model.load(best_model_path, device)
            improved = evaluator.eval(best_model, model)
            
            if improved:
                model.save(best_model_path)
                
            plot_timeseries(model.elo_history, args.figure_path, 'episode', 'elo', 'Elo')
            
        env.reset()

if __name__ == "__main__":
    main()