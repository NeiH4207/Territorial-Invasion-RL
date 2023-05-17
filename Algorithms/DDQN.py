from collections import deque
from copy import deepcopy
import numpy as np
from torch import optim as optim
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
from tqdm import tqdm

from Algorithms.DQN import DQN

class DDQN(DQN):
    def __init__(self, n_observations=None, n_actions=None, model=None,
                    optimizer='adam', lr=0.001, tau=0.005, gamma=0.99,
                    epsilon=0.9, epsilon_min=0.05, epsilon_decay=0.99,
                    memory_size=4096,  model_path=None):
        super().__init__(n_observations, n_actions, model, optimizer, 
                         lr, tau, gamma, epsilon, epsilon_min, 
                         epsilon_decay, memory_size, model_path)
        
    def replay(self, batch_size, verbose=False):

        if len(self.memory) < batch_size:
            return 0 # memory is still not full
        if verbose:
            _tqdm = tqdm(range(len(self.memory) // batch_size + 1), desc='Replay')
        else:
            _tqdm = range(len(self.memory) // batch_size + 1)
        total_loss = 0
        
        for i in _tqdm:
            minibatch = random.sample(self.memory, batch_size)
            state_batch = []
            action_batch = []
            next_state_batch = []
            reward_batch = []
            done_batch = []
            
            for state, action, next_state, reward, done in minibatch:
                state_batch.append(torch.Tensor(state).to(self.device))
                action_batch.append(torch.LongTensor([action]).to(self.device))
                next_state_batch.append(torch.Tensor(next_state).to(self.device))
                reward_batch.append(torch.Tensor([reward]).to(self.device))
                done_batch.append(torch.Tensor([done]).to(self.device))
                
            
            next_state_values = torch.zeros(batch_size, device=self.device)
            with torch.no_grad():
                next_action_batch = self.policy_net(torch.stack(next_state_batch, axis=0)).max(1)[1]
                next_state_values = self.target_net(torch.stack(next_state_batch, axis=0))
                next_state_values = next_state_values.gather(1, next_action_batch.reshape(-1, 1)).squeeze()
                
                
            state_batch = torch.stack(state_batch, axis=0)
            action_batch = torch.cat(action_batch)
            reward_batch = torch.cat(reward_batch)
            done_batch = torch.cat(done_batch)
            
            expected_state_action_values = (1 - done_batch) * (next_state_values * self.gamma) + reward_batch

            state_action_values = self.policy_net(state_batch).gather(1, action_batch.reshape(-1, 1))
            
            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
                # Optimize the model
            self.policy_net.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.policy_net.optimizer.step()
        
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
            self.target_net.load_state_dict(target_net_state_dict)
            total_loss += loss.item()
            mean_loss = total_loss / (i + 1)
        self.history['loss'].append(mean_loss)
        