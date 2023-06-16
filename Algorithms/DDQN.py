import numpy as np
from torch import optim as optim
import random
import torch
import torch.nn as nn
from tqdm import tqdm

from Algorithms.DQN import DQN

class DDQN(DQN):
    def __init__(self, n_observations=None, n_actions=None, model=None,
                    tau=0.005, gamma=0.99, epsilon=0.9, epsilon_min=0.05, 
                    epsilon_decay=0.99,
                    memory_size=4096,  model_path=None):
        super().__init__(n_observations, n_actions, model, tau, 
                         gamma, epsilon, epsilon_min, epsilon_decay,
                         memory_size, model_path)
        
    def get_action(self, state, valid_actions=None):
        state = torch.FloatTensor(np.array(state)).to(self.device)
        act_values = self.policy_net.predict(state)[0]
        # set value of invalid actions to -inf
        if valid_actions is not None:
            act_values[~valid_actions] = -float('inf')
        return int(np.argmax(act_values))  # returns action
        
    def replay(self, batch_size, verbose=False):

        if len(self.memory) < batch_size:
            return 0 # memory is still not full
        if verbose:
            _tqdm = tqdm(range(len(self.memory) // batch_size + 1), desc='Replay')
        else:
            _tqdm = range(len(self.memory) // batch_size + 1)
        total_loss = 0
        self.policy_net.train()
        
        for i in _tqdm:
            minibatch = self.memory.sample(batch_size)
            state_batch = torch.Tensor(minibatch[0]).to(self.device) 
            action_batch = torch.LongTensor(minibatch[1]).to(self.device)
            reward_batch = torch.Tensor(minibatch[2]).to(self.device)
            next_state_batch =  torch.Tensor(minibatch[3]).to(self.device)
            done_batch = torch.Tensor(minibatch[4]).to(self.device)
                
            
            next_state_values = torch.zeros(batch_size, device=self.device)
            with torch.no_grad():
                next_action_batch = self.policy_net(next_state_batch).max(1)[1]
                next_state_values = self.target_net(next_state_batch)
                next_state_values = next_state_values.gather(1, next_action_batch.reshape(-1, 1)).squeeze()
            
            expected_state_action_values = (1 - done_batch) * (next_state_values.unsqueeze(1) * self.gamma) + reward_batch

            state_action_values = self.policy_net(state_batch).gather(1, action_batch.reshape(-1, 1))
            
            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values)
                # Optimize the model
            self.policy_net.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.policy_net.optimizer.step()
            total_loss += loss.item()
            mean_loss = total_loss / (i + 1)
            
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

        self.policy_net.add_loss(mean_loss)
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        return self.policy_net.get_loss()