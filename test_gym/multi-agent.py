import time
import gym

from models.GYM.GymDQN import CartPole
"""
Created on Tue Apr 27 2:19:47 2023
@author: hien
"""
from itertools import count
import logging
import os
import torch
from tqdm import tqdm
from src.utils import *
log = logging.getLogger(__name__)
from argparse import ArgumentParser

from algorithms.Rainbow import Rainbow
from models.RainbowNet import RainbowNet
import gym

def argument_parser():
    parser = ArgumentParser()
    # Game options
    parser.add_argument('--show-screen', type=bool, default=True)
    parser.add_argument('--render-last', type=bool, default=True)
    parser.add_argument('--figure-path', type=str, default='figures/')
    
    # DDQN arguments
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.6)
    parser.add_argument('--prior_eps', type=float, default=1e-6)
    
    parser.add_argument('--n-step', type=int, default=3)
    
    # model training arguments
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--memory-size', type=int, default=8192)
    parser.add_argument('--num-episodes', type=int, default=1000)
    parser.add_argument('--model-path', type=str, default='tmp/model.pt')
    parser.add_argument('--load-model', action='store_true')
    
    return parser.parse_args()

def main():
    args = argument_parser()
    if args.show_screen:
        mode = 'human'
    else:
        mode = 'rgb_array'
        
    env = gym.make('ma_gym:Combat-v0')
    n_actions = env.action_space[0].n
    observation_shape = env._grid_shape[0] * 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    algorithm = None
    set_seed(1)
    
    model = CartPole(
        observation_shape=observation_shape,
        n_actions=n_actions,
        atom_size=51, 
        v_min=-20,
        v_max=20,
        optimizer=args.optimizer,
        lr=args.lr,
        device=device
    )
    
    model = model.to(device)
    
    algorithm = Rainbow(   
        observation_shape=observation_shape, 
        n_actions=n_actions,
        model=model,
        tau=args.tau,
        gamma=args.gamma,
        memory_size=args.memory_size,
        model_path=args.model_path,
        batch_size=args.batch_size,
        alpha=args.alpha,
        beta=args.beta,
        prior_eps=args.prior_eps,
        n_step=args.n_step,
        v_min=0,
        v_max=500
    )
    algorithm.set_multi_agent_env(1, env.n_agents)
    if args.model_path:
        model_dir = os.path.dirname(args.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logging.info('Created model directory: {}'.format(model_dir))
        if args.load_model:
            algorithm.load_model(args.model_path)
    
    args.figure_path = os.path.join(args.figure_path, 'Rainbow')
    
    if not os.path.exists(args.figure_path):
        os.makedirs(args.figure_path)
        
    print("History and ep_rewards saved at {}".format(args.figure_path))
        
    ep_rewards = []
    
    for episode in tqdm(range(args.num_episodes)):
        obs_n = env.reset()
        env.render()
        cnt = 0
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0
        while not all(done_n):
            actions = [algorithm.get_action(obs_i) for obs_i in obs_n]
            next_obs_n, reward_n, done_n, info = env.step(actions)
            env.render()
            ep_reward += sum(reward_n)
            for state, action, reward, next_state, done in zip(obs_n, actions, reward_n, obs_n, done_n):
                # reward = np.mean(reward_n) * 0.5 + reward * 0.5
                transition = [state, action, reward, next_state, done]
                one_step_transition = algorithm.memory_n.store(*transition)
                if one_step_transition:
                    algorithm.memory.store(*one_step_transition)
                algorithm.memorize(state, action, reward, next_state, done)
            obs_n = next_obs_n
        
        ep_rewards.append(ep_reward)
        
        if episode % 3 == 0 and algorithm.fully_mem(0.25):
            history_loss = algorithm.replay(args.batch_size)
            plot_timeseries(history_loss, args.figure_path, 'episode', 'loss', 'Training Loss')
            plot_timeseries(ep_rewards, args.figure_path, 'episode', 'ep_rewards', 'Training ep_rewards')
            if ep_rewards[-1] >= max(ep_rewards[:-1]):
                algorithm.save_model()
                
    if args.render_last:
        algorithm.load_model(args.model_path, device)
        env = gym.make('ma_gym:Combat-v0')
        
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