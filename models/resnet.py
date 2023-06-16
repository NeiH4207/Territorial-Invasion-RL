#!/usr/bin/env python
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim
from models import ModelConfig
import logging

from models.NoisyLayer import NoisyLinear
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


