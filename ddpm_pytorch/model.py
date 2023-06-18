import math
from typing import Optional, Tuple, Union, List

import torch 
from torch import nn

class Swish(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self):
        pass
    def forward(self, t: torch.Tensor):
        return emb 
