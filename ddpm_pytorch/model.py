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
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)  # (half_dim,)
        emb = t[:, None] * emb[None, :] # (bs, half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=1) # (bs, 2*half_dim)
        emb = self.act(self.lin1(emb)) # (bs, n_channels)
        emb = self.lin2(emb) # (bs, n_channels)
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

        self.middle = MiddleBlock(out_channels, n_channels * 4, )

        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels*4, is_attn[i]))
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels*4, is_attn[i]))
            in_channels = out_channels

            if i > 0:
                up.append(UpSample(in_channels))
        
        self.up = nn.ModuleList(up)
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x: (bs, in_channels, h, w)
        t: (bs, )
        """
        t = self.time_emb(t)
        x = self.image_proj(x)
        h = [x]

        for m in self.down:
            x = m(x, t)
            h.append(x)
        
        x = self.middle(x, t)

        for m in self.up:
            if isinstance(m, UpSample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)
        
        return self.final(self.act(self.norm(x)))
