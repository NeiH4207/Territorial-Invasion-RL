from collections import deque
import torch as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim as optim
import random
import numpy as np
from collections import deque

class AgentNet(nn.Module):
    
    def __init__(self) -> None:
        super(AgentNet, self).__init__()
        pass
    
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
        
    def predict(self, s):
        s = torch.FloatTensor(np.array(s)).to(self.device).detach()
        with torch.no_grad():
            self.eval()
            pi, v = self.forward(s)
            return torch.exp(pi).cpu().numpy()[0], v.item()
    
    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr    
            
    def reset_grad(self):
        self.optimizer.zero_grad()
        
    def step(self):
        self.optimizer.step()
        
    def optimize(self):
        self.optimizer.step()
        
    def loss_pi(self, targets, outputs):
        return -torch.mean(torch.sum(targets * outputs, 1))

    def loss_v(self, targets, outputs):
        return F.mse_loss(outputs.view(-1), targets)
          
    def save_train_losses(self, train_losses):
        self.train_losses = train_losses
        