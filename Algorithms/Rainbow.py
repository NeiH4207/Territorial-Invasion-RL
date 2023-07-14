from typing import Dict
import numpy as np
from torch import optim as optim
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm

from Algorithms.DQN import DQN
from src.priority import PrioritizedReplayBuffer, ReplayBuffer

class Rainbow(DQN):
    
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
        # N-step Learning
        n_step: int = 3,
        ):

        super().__init__(n_observations, n_actions, model, tau, 
                         gamma, epsilon, epsilon_min, epsilon_decay,
                         memory_size, model_path)
        self.memory = self.memory = PrioritizedReplayBuffer(
            n_observations, memory_size, batch_size, alpha
        )
        self.beta = beta
        self.prior_eps = prior_eps
        self.n_step = n_step
        self.transition = list()
        self.memory_n = ReplayBuffer(
                n_observations, memory_size, batch_size, n_step=n_step, gamma=gamma
            )
    
    def reset_memory(self):        
        self.memory.size = 0
        
    def calculate_dqn_loss(
        self, 
        samples: Dict[str, np.ndarray], 
        gamma: float
    ) -> torch.Tensor:
            state_batch = Variable(torch.FloatTensor(samples['obs'])).to(self.device)
            action_batch = Variable(torch.LongTensor(samples['acts'])).to(self.device)
            next_state_batch = Variable(torch.FloatTensor(samples['next_obs'])).to(self.device)
            reward_batch = Variable(torch.FloatTensor(samples['rews'])).to(self.device)
            done_batch = Variable(torch.FloatTensor(samples['done'])).to(self.device)
            
            next_state_values = torch.zeros(samples['next_obs'].shape[0], device=self.device)
            with torch.no_grad():
                next_action_batch = self.policy_net(next_state_batch).max(1)[1]
                next_state_values = self.target_net(next_state_batch)
                next_state_values = next_state_values.gather(1, next_action_batch.reshape(-1, 1)).squeeze()
                
            expected_state_action_values = (1 - done_batch) * (next_state_values * gamma) + reward_batch

            state_action_values = self.policy_net(state_batch).gather(1, action_batch.reshape(-1, 1))
            
            # Compute Huber loss
            elementwise_loss = F.smooth_l1_loss(state_action_values, 
                                                expected_state_action_values.unsqueeze(1), 
                                                reduction="none")
            return elementwise_loss
    
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
            samples = self.memory.sample_batch(self.beta)
            weight_batch = Variable(torch.FloatTensor(samples['weights'])).to(self.device)
            elementwise_loss = self.calculate_dqn_loss(samples, self.gamma)
            loss = torch.mean(elementwise_loss * weight_batch)
            
            n_step_samples = self.memory_n.sample_batch_from_idxs(samples['indices'])
            gamma = self.gamma ** self.n_step
            n_elementwise_loss_loss = self.calculate_dqn_loss(n_step_samples, gamma)
            n_loss = torch.mean(n_elementwise_loss_loss * weight_batch)
            loss += n_loss
                # Optimize the model
            self.policy_net.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.policy_net.optimizer.step()
            # PER: update priorities
            loss_for_prior = elementwise_loss.detach().cpu().numpy()
            new_priorities = loss_for_prior + self.prior_eps
            self.memory.update_priorities(samples['indices'], new_priorities)
            
            self.policy_net.reset_noise()
            self.target_net.reset_noise()
            self.soft_update()
            total_loss += loss.item()
            mean_loss = total_loss / (i + 1)
            
        self.policy_net.add_loss(mean_loss)
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        
        return self.policy_net.get_loss()
        
    
    def update_beta(self):
        self.beta = min(1.0, self.beta + 1e-4)
        