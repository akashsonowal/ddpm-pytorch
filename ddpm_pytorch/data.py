#!usr/bin/python3
import torchvision


class MNISTDataset(torchvision.datasets.MNIST):
    def __init__(self, image_size):
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(image_size),
                torchvision.transforms.ToTensor(),
            ]
        )
        super().__init__(root="../data", train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]  # get only the image