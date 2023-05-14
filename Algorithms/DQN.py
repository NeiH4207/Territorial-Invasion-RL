from collections import deque
from copy import deepcopy
import numpy as np
from torch import optim as optim
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn

class DQN():
    def __init__(self, n_observations, n_actions, model, optimizer, learning_rate, model_path=None):
        super().__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.memory = deque(maxlen=8096)
        self.gamma = 0.99
        self.epsilon = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_net = model
        self.target_net = deepcopy(model)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.set_optimizer(optimizer, learning_rate)  
        self.model_path = model_path
        self.history = {
            'loss': [],
            'reward': []
        }
    
    def memorize(self, state, action, next_state, reward, done):
        # storage
        self.memory.append((state, action, next_state, reward, done))
        
    def get_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.rand() <= epsilon:
            return random.randrange(self.n_actions)
        state = torch.FloatTensor(np.array(state)).to(self.device)
        act_values = self.policy_net.predict(state)
        return int(np.argmax(act_values))  # returns action
        
    def replay(self, batch_size):

        if len(self.memory) < batch_size:
            return 0 # memory is still not full
        
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
            next_state_values = self.target_net(torch.stack(next_state_batch, axis=0)).max(1)[0]
            
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
        TAU = 0.005
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)
        self.history['loss'].append(loss.item())
        return self.history
        
    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def load_model(self, path):
        self.policy_net.load(path)
        self.target_net = deepcopy(self.policy_net)
        
    def save_model(self, path):
        self.policy_net.save(path)