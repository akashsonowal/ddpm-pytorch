from typing import List

from tqdm import tqdm

import torch
import torch.utils.data

from ddpm_pytorch import MNISTDataset, UNet, DenoiseDiffusion


def train_one_epoch(diffusion, optimizer, data_loader, device):
    for data in tqdm(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss = diffusion.loss(data)
        loss.backward()
        optimizer.step()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = MNISTDataset(image_size=32)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, pin_memory=True
    )

    eps_model = UNet(
        image_channels=1,
        n_channels=64,
        ch_mults=[1, 2, 2, 4],
        is_attn=[False, False, False, True],
    ).to(device)
    diffusion = DenoiseDiffusion(eps_model, n_steps=10, device=device)

    optimizer = torch.optim.Adam(eps_model.parameters(), lr=2e-5)
    epochs = 1  # 1000

    for epoch in tqdm(range(epochs)):
        train_one_epoch(diffusion, optimizer, data_loader, device)

    torch.save(diffusion, "ddpm_model.pth")


if __name__ == "__main__":
    main()
