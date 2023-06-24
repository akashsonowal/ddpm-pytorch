from typing import Tuple, Optional

import torch 
import torch.nn.functional as F
import torch.utils.data
from torch import nn

def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)