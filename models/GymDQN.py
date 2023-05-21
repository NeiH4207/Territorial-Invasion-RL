#!/usr/bin/env python
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim

from models.NoisyLayer import NoisyLinear


class GymDQN(nn.Module):

    def __init__(
        self, 
        n_observations: int, 
        n_actions: int, 
        optimizer: str = "adamw",
        lr: float = 0.001) -> None:
        super(GymDQN, self).__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.value = NoisyLinear(128, n_actions)
        self.advance = NoisyLinear(128, n_actions)
        self.set_optimizer(optimizer, lr)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        value = self.value(x) 
        adv = self.advance(x) 
        mean_adv = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - mean_adv
        return Q 
        
    def set_loss_function(self, loss):
        if loss == "mse":
            self.loss = nn.MSELoss()		# Hàm loss là tổng bình phương sai lệch
        elif loss == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()
        elif loss == "bce":
            self.loss = nn.BCELoss()		# Hàm loss là binary cross entropy, với đầu ra 2 lớp
        elif loss == "bce_logits":
            self.loss = nn.BCEWithLogitsLoss()		# Hàm BCE logit sau đầu ra dự báo có thêm sigmoid, giống BCE
        elif loss == "l1":
            self.loss = nn.L1Loss()
        elif loss == "smooth_l1":
            self.loss = nn.SmoothL1Loss()		# Hàm L1 loss nhưng có đỉnh được làm trơn, khả vi với mọi điểm
        elif loss == "soft_margin":
            self.loss = nn.SoftMarginLoss()		# Hàm tối ưu logistic loss 2 lớp của mục tiêu và đầu ra dự báo
        else:
            raise ValueError("Loss function not found")
        
    def predict(self, x: torch.Tensor) -> np.ndarray:
        x = x.reshape(-1, self.n_observations)
        output = self.forward(x)
        return output.detach().cpu().numpy()
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.advance.reset_noise()
        self.value.reset_noise()
    
    def set_optimizer(self, optimizer, lr):
        if optimizer == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=lr)		# Tối ưu theo gradient descent thuần túy
        elif optimizer == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == "adadelta":
            self.optimizer = optim.Adadelta(self.parameters(), lr=lr)		# Phương pháp Adadelta có lr update
        elif optimizer == "adagrad":
            self.optimizer = optim.Adagrad(self.parameters(), lr=lr)		# Phương pháp Adagrad chỉ cập nhật lr ko nhớ
        elif optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        else:
            self.optimizer = optim.AdamW(self.parameters(), lr=lr, amsgrad=True)
    
    def save(self, path=None):
        torch.save(self.state_dict(), path)
        print("Model saved at {}".format(path))
        
    def load(self, path=None):
        if path is None:
            raise ValueError("Path is not defined")
        self.load_state_dict(torch.load(path, map_location=torch.device(self.device)))
        print('Model loaded from {}'.format(path))
