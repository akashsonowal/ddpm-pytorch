from typing import List 

from tqdm import tqdm

import torch 
import torch.utils.data
import torchvision
from PIL import Image 

from torch.utils.data import Dataset
from ddpm_pytorch.model import UNet
from ddpm_pytorch.ddpm import DenoiseDiffusion

device = "cuda" if torch.cuda.is_available() else "cpu"



class CelebADataset(Dataset):
    def __init__(self, image_size: int):
        super().__init__()
    
    def __getitem__(self, index: int):
        img = Image.open(self._files[index])
        return self._transform(img)

def train_one_epoch():
    for data in tqdm(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss = diffusion.loss(data)
        loss.backward()
        optimizer.step()

def sample():
    pass 

def main():
    for epoch in tqdm(range(epochs)):
        train_one_epoch()
        sample()


if __name__ == "__main__":
    main()