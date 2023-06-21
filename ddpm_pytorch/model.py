import math
from typing import Optional, Tuple, Union, List

import torch 
from torch import nn

class Swish(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)        

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)  # [half_dim]
        emb = t[:, None] * emb[None, :] # (bs, half_dim, n_channels)
        emb = torch.cat((emb.sin(), emb.cos()), dim=1) # (bs, 2*half_dim. n_channels)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb 

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32, dropout: float = 0.1):
        super().__init__()
        pass 
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        pass 

class AttentionBlock(nn.Module):
    pass

class DownBlock(nn.Module):
    pass

class UpBlock(nn.Module):
    pass 

class MiddleBlock(nn.Module):
    pass 

class UpSample(nn.Module):
    pass 

class DownSample(nn.Module):
    pass 

class UNet(nn.Module):
    def __init__(self, image_channels: int = 3, n_channels: int = 64, ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 3, 4),  is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True), n_blocks: int = 2):
        super().__init__()
        n_resolutions = len(ch_mults)
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))
        self.time_emb = TimeEmbedding(n_channels*4)

        down = []

        out_channels = in_channels = n_channels

        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]

            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels*4, is_attn[i]))
                in_channels = out_channels
            
            if i < n_resolutions - 1:
                down.append(DownSample(in_channels))
        
        self.down = nn.ModuleList(down)
        