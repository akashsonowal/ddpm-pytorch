from typing import List 

from torch.utils.data import Dataset
from ddpm_pytorch.model import UNet

class CelebADataset(Dataset):
    def __init__(self, image_size: int):
        super().__init__()
    
    def __getitem__(self, index: int):
        img = Image.open(self._files[index])
        return self._transform(img)

def main():
    pass

if __name__ == "__main__":
    main()