"""
Created on Tue Apr 27 2:19:47 2023
@author: hien
"""
from __future__ import division
import json
import logging
import time
from Algorithms.RandomStep import RandomStep
from src.environment import AgentFighting
log = logging.getLogger(__name__)
from random import seed
from argparse import ArgumentParser

def argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--show-screen', type=bool, default=True)
    parser.add_argument('--render', type=bool, default=True)
    return parser.parse_args()

def main():
    args = argument_parser()
    configs = json.load(open('config.json'))
    env = AgentFighting(args, configs, args.show_screen)
    algorithm = RandomStep(n_actions=env.n_actions, num_agents=env.num_agents)
    state = env.reset()
    while not env.game_ended():
        action = algorithm.get_action(state)
        _ = env.step(action)
        env.render()
        time.sleep(0.05)
    
    winner = env.get_winner()
    if winner == -1:
        logging.info('Game ended. Draw')
    else:
        logging.info('Game ended. Winner: {}'.format(env.get_winner()))
        
    time.sleep(3)

if __name__ == "__main__":
    main()