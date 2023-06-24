import numpy as np
import torch 
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image, resize 
from ddpm_pytorch.ddpm import DenoiseDiffusion
from ddpm_pytorch.utils import gather

class Sampler:
    def __init__(self, diffusion: DenoiseDiffusion, image_channels: int, image_size: int, device: torch.device):
        self.device = device
        self.image_size = image_size
        self.image_channels = image_channels
        self.diffusion = diffusion
        self.n_steps = diffusion.n_steps
        self.beta = diffusion.beta
        self.alpha = diffusion.alpha
        self.alpha_bar = diffusion.alpha_bar
        alpha_bar_tm1 = torch.cat([self.alpha_bar.new_ones((1,)), self.alpha_bar[:-1]])
        self.beta_tilde = self.beta * (1 - alpha_bar_tm1) / (1 - self.alpha_bar)
        self.mu_tilde_coef2 = (self.alpha ** 0.5) * (1 - alpha_bar_tm1) / (1 - self.alpha_bar)
        self.sigma2 = self.beta 

    def sample(self, ):
        pass 

def main():
    pass 

if __name__ == "__main__":
    main()