from collections import deque
from copy import deepcopy
import numpy as np
from torch import optim as optim
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm

from Algorithms.DQN import DQN
from src.priority import PrioritizedReplayBuffer

class PER(DQN):
    
    def __init__(
        self, 
        n_observations=None,
        n_actions=None, 
        model=None,
        # DQN parameters
        tau=0.005, 
        gamma=0.99, 
        epsilon=0.9, 
        epsilon_min=0.05, 
        epsilon_decay=0.99, 
        memory_size=4096, 
        batch_size=32, 
        model_path=None,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        ):

        super().__init__(n_observations, n_actions, model, tau, 
                         gamma, epsilon, epsilon_min, epsilon_decay,
                         memory_size, model_path)
        self.memory = self.memory = PrioritizedReplayBuffer(
            n_observations, memory_size, batch_size, alpha
        )
        self.beta = beta
        self.prior_eps = prior_eps
        
    def replay(self, batch_size, verbose=False):

        if len(self.memory) < batch_size:
            return 0 
        
        if verbose:
            _tqdm = tqdm(range(len(self.memory) // batch_size + 1), desc='Replay')
        else:
            _tqdm = range(len(self.memory) // batch_size + 1)
        total_loss = 0
        self.policy_net.train()
        
        for i in _tqdm:
            minibatch = self.memory.sample_batch(self.beta)
            state_batch = Variable(torch.FloatTensor(minibatch['obs'])).to(self.device)
            action_batch = Variable(torch.LongTensor(minibatch['acts'])).to(self.device)
            next_state_batch = Variable(torch.FloatTensor(minibatch['next_obs'])).to(self.device)
            reward_batch = Variable(torch.FloatTensor(minibatch['rews'])).to(self.device)
            done_batch = Variable(torch.FloatTensor(minibatch['done'])).to(self.device)
            weight_batch = Variable(torch.FloatTensor(minibatch['weights'])).to(self.device)
            indice_batch = minibatch['indices']
            
            next_state_values = torch.zeros(batch_size, device=self.device)
            with torch.no_grad():
                next_action_batch = self.policy_net(next_state_batch).max(1)[1]
                next_state_values = self.target_net(next_state_batch)
                next_state_values = next_state_values.gather(1, next_action_batch.reshape(-1, 1)).squeeze()
                
            expected_state_action_values = (1 - done_batch) * (next_state_values * self.gamma) + reward_batch

            state_action_values = self.policy_net(state_batch).gather(1, action_batch.reshape(-1, 1))
            
            # Compute Huber loss
            elementwise_loss = F.smooth_l1_loss(state_action_values, 
                                                expected_state_action_values.unsqueeze(1), 
                                                reduction="none")
            loss = torch.mean(elementwise_loss * weight_batch)
                # Optimize the model
            self.policy_net.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.policy_net.optimizer.step()
            
            # PER: update priorities
            loss_for_prior = elementwise_loss.detach().cpu().numpy()
            new_priorities = loss_for_prior + self.prior_eps
            self.memory.update_priorities(indice_batch, new_priorities)
            self.soft_update()
            total_loss += loss.item()
            mean_loss = total_loss / (i + 1)
            
        self.policy_net.add_loss(mean_loss)
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        
        if self.counter % int(1 / self.tau) == 0:
            self.hard_update()
            
        return self.policy_net.get_loss()
        