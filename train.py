from typing import List 

from tqdm import tqdm

import torch 
import torch.utils.data
import torchvision
from PIL import Image 

from pathlib import Path

from torch.utils.data import Dataset

from ddpm_pytorch import MNISTDataset, UNet, DenoiseDiffusion

def train_one_epoch(diffusion, optimizer, data_loader, device):
    for data in tqdm(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss = diffusion.loss(data)
        loss.backward()
        optimizer.step()

def sample(diffusion, n_samples, image_channels, image_size, n_steps, device):
    with torch.no_grad():
        x = torch.randn([n_samples, image_channels, image_size, image_size], device)

        for t_ in range(n_steps):
            t = n_steps - t_- 1
            x = diffusion.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = MNISTDataset(image_size=32)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)

    eps_model = UNet(image_channels=1, n_channels=64, ch_mults=[1, 2, 2, 4], is_attn=[False, False, False, True]).to(device)
    diffusion = DenoiseDiffusion(eps_model, n_steps=1000, device=device)

    optimizer = torch.optim.Adam(eps_model.parameters(), lr=2e-5)
    epochs = 10 # 1000

    for epoch in tqdm(range(epochs)):
        train_one_epoch(diffusion, optimizer, data_loader, device)
        sample(diffusion, n_samples=16, image_channels=3, image_size=32, n_steps=1000, device=device)

if __name__ == "__main__":  
    main()
