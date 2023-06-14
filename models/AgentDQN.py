#!/usr/bin/env python
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim
from models import ModelConfig
import logging
from torchvision import models

from models.NoisyLayer import NoisyLinear
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

class OutBlock(nn.Module):
    def __init__(self, config, input_shape, output_shape, dueling=True):
        super(OutBlock, self).__init__()
        self.config = config
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        self.fc1 = nn.Linear(input_shape, config['fc1-num-units'])
        self.fc_bn1 = nn.BatchNorm1d(config['fc1-num-units'])

        self.fc2 = nn.Linear(config['fc1-num-units'], config['fc2-num-units'])
        self.fc_bn2 = nn.BatchNorm1d(config['fc2-num-units'])
        
        self.dueling = dueling
        if self.dueling:
            self.value = NoisyLinear(config['fc2-num-units'], output_shape)
            self.advance = NoisyLinear(config['fc2-num-units'], output_shape)
        else:
            self.Qvalue = nn.Linear(config['fc2-num-units'], output_shape)
    
    def forward(self, s):
        out = F.dropout(F.relu(self.fc1(s)), p=self.config['fc1-dropout'], training=self.training)  # batch_size x 1024
        out = F.relu(self.fc2(out))
        if self.dueling:
            value = self.value(out) 
            adv = self.advance(out) 
            mean_adv = torch.mean(adv, dim=1, keepdim=True)
            Qvalue = value + adv - mean_adv
        else:
            Qvalue = self.Qvalue(out) 
        return Qvalue
    
    
class DQN(nn.Module):
    def __init__(self, input_shape, output_shape, optimizer='adamw', lr=0.001, dueling=True):
        super(DQN, self).__init__()
        # game params
        self.name = 'nnet9x9'
        config = ModelConfig[self.name]
        self.config = config
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.in_channels = input_shape[0]
        self.elo_history = np.array([1000])
        self.loss_history = np.array([])
        
        self.resnet = models.resnet18(pretrained=False)
        self.flatten_dim = 1000
        self.outblock = OutBlock(config, self.flatten_dim, output_shape, dueling=dueling)
        
        self.set_optimizer(optimizer, lr)
    
    def forward(self, s):
        s = self.resnet(s)
        s = s.view(-1, self.flatten_dim)
        s = self.outblock(s)
        return s
    
    def get_elo(self):
        return self.elo_history[-1]
    
    def set_elo(self, ELO):
        self.elo_history = np.append(self.elo_history, ELO)
    
    def add_loss(self, loss):
        self.loss_history = np.append(self.loss_history, loss)
    
    def get_loss(self):
        return self.loss_history
    
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
    
    def predict(self, x):	# Chuyển đầu ra x về dạng torch tensor
        self.eval()
        x = x.reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        output = self.forward(x).detach()
        return output.cpu().data.numpy()
    
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
        if path is None:
            logging.error('Model path not specified')
        state_dict = self.state_dict()
        checkpoint = {
            'elo_history': self.elo_history,
            'loss_history': self.loss_history,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        logging.info('Model saved to {}'.format(path))
        
    def load(self, path=None, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if path is None:
            raise ValueError("Path is not defined")
        checkpoint = torch.load(path, map_location=device)
        self.elo_history = checkpoint['elo_history']
        self.loss_history = checkpoint['loss_history']
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('Model loaded from {}'.format(path))

    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.outblock.advance.reset_noise()
        self.outblock.value.reset_noise()