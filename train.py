from typing import List 

from tqdm import tqdm

import torch 
import torch.utils.data
import torchvision
from PIL import Image 

from pathlib import Path

from torch.utils.data import Dataset
from ddpm_pytorch.model import UNet
from ddpm_pytorch.ddpm import DenoiseDiffusion

class CelebADataset(Dataset):
    def __init__(self, image_size: int):
        super().__init__()
        folder = Path("./data") / "celebA"
        self._files = [p for p in folder.glob(f"**/*.jpg")]
        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self._files)
    
    def __getitem__(self, index: int):
        img = Image.open(self._files[index])
        return self._transform(img)

def train_one_epoch(diffusion, optimizer, data_loader, device):
    for data in tqdm(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss = diffusion.loss(data)
        loss.backward()
        optimizer.step()

def sample():
    pass 


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = CelebADataset(image_size=32)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)

    eps_model = UNet(image_channels=3, n_channels=64, ch_mults=[1, 2, 2, 4], is_attn=[False, False, False, True]).to(device)
    diffusion = DenoiseDiffusion(eps_model, n_steps=1000, device=device)

    optimizer = torch.optim.Adam(eps_model.parameters(), lr=2e-5)

    for epoch in tqdm(range(epochs)):
        train_one_epoch(diffusion, optimizer, data_loader, device)
        sample()


if __name__ == "__main__":  
    main()