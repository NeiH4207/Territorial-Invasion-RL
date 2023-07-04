"""
Created on Tue Apr 27 2:19:47 2023
@author: hien
"""
from __future__ import division
from itertools import count
import logging
import os
import time
import torch
from tqdm import tqdm
from src.utils import *
log = logging.getLogger(__name__)
from argparse import ArgumentParser
from Algorithms.DQN import DQN
from models.GYM.GymDQN import GymDQN
import gym

def argument_parser():
    parser = ArgumentParser()
    # Game options
    parser.add_argument('--show-screen', type=bool)
    parser.add_argument('--figure-path', type=str, default='figures/')
    
    # DDQN arguments
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=int, default=0.01)
    
    # model training arguments
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--memory-size', type=int, default=4096)
    parser.add_argument('--num-episodes', type=int, default=2000)
    parser.add_argument('--model-path', type=str, default='tmp/model.pt')
    parser.add_argument('--load-model', action='store_true')
    
    return parser.parse_args()

def main():
    args = argument_parser()
    if args.show_screen:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')
        
    n_observations, n_actions = env.observation_space.shape[0], env.action_space.n
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    algorithm = None
    set_seed(1)
    
    model = GymDQN(
        n_observations=n_observations,
        n_actions=n_actions,
        optimizer=args.optimizer,
        lr=args.lr,
    ).to(device)
    
    algorithm = DQN(   n_observations=n_observations, 
                        n_actions=n_actions,
                        model=model,
                        tau=args.tau,
                        gamma=args.gamma,
                        memory_size=args.memory_size,
                        model_path=args.model_path
                    )
        
    if args.model_path:
        model_dir = os.path.dirname(args.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logging.info('Created model directory: {}'.format(model_dir))
        if args.load_model:
            algorithm.load_model(args.model_path)
    
    args.figure_path = os.path.join(args.figure_path, 'DQN')
    
    if not os.path.exists(args.figure_path):
        os.makedirs(args.figure_path)
        
    print("History and timesteps saved at {}".format(args.figure_path))
        
    timesteps = []
        
    for episode in tqdm(range(args.num_episodes)):
        done = False
        state, info = env.reset()
        cnt = 0
        for cnt in count():
            env.render()
            action = algorithm.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            reward = reward if not done else -1
            algorithm.memorize(state, action, reward, next_state, done)
            state = next_state
            if done or truncated:
                break
            
        timesteps.append(cnt)
        
        if episode % 3 == 0 and algorithm.fully_mem(0.25):
            history_loss = algorithm.replay(args.batch_size)
            plot_timeseries(history_loss, args.figure_path, 'episode', 'loss', 'Training Loss')
            plot_timeseries(timesteps, args.figure_path, 'episode', 'timesteps', 'Training Timesteps')
            if cnt >= max(timesteps):
                algorithm.save_model()
                
    if args.render_last:
        algorithm.load_model(args.model_path, device)
        env = gym.make('CartPole-v1', render_mode='human')
        
        done = False
        state, info = env.reset()
        cnt = 0
        for cnt in count():
            env.render()
            action = algorithm.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            if done or truncated:
                break

if __name__ == "__main__":
    main()