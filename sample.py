import os
import numpy as np
import torch 
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image, resize 
from ddpm_pytorch.ddpm import gather, DenoiseDiffusion 
from ddpm_pytorch import MNISTDataset

class Sampler:
    def __init__(self, diffusion: DenoiseDiffusion, image_channels: int, image_size: int, device: torch.device):
        self.device = device
        self.image_size = image_size
        self.image_channels = image_channels
        self.diffusion = diffusion
        self.n_steps = diffusion.n_steps
        self.eps_model = diffusion.eps_model
        self.beta = diffusion.beta
        self.alpha = diffusion.alpha
        self.alpha_bar = diffusion.alpha_bar
        self.sigma2 = self.beta 
    
    def _sample_x0(self, xt: torch.Tensor, n_steps: int): # reverse diffusion to get x0 from xt
        n_samples = xt.shape[0]

        for t_ in range(n_steps):
            t = n_steps - t_ - 1
            xt = self.diffusion.p_sample(xt, xt.new_full((n_samples,), t, dtype=torch.long))
        return xt
    
    def sample(self, n_samples: int = 5): # reverse diffusion
        xt = torch.randn([n_samples, self.image_channels, self.image_size, self.image_size], device=self.device)
        x0 = self._sample_x0(xt, self.n_steps)

        for i in range(n_samples):
            self.show_image(x0[i], f"generated image {i}")
    
    def show_image(self, img, title=""):
        img = img.clip(0, 1)
        img = img.detach().cpu().numpy()
        plt.imshow(img.transpose(1, 2, 0))
        plt.title(title)
        plt.show()
    
    def make_video(self, frames, path="video.mp4"):
        import imageio
        writer = imageio.get_writer(path, fps=len(frames) // 20)

        for f in frames:
            f = f.clip(0, 1)
            f = to_pil_image(resize(f, [368, 368]))
            writer.append_data(np.array(f))

        writer.close()
    
    def sample_animation(self, n_samples: int = 1, n_frames: int = 1000, create_video: bool = True):
        xt = torch.randn([n_samples, self.image_channels, self.image_size, self.image_size], device=self.device)
        
        if self.n_steps == n_frames:
            interval = self.n_steps // n_frames # 1000 // 1000 = 1
        else:
            interval = 1

        frames = []

        for t_inv in range(self.n_steps):
            t_ = self.n_steps - t_inv - 1
            t = xt.new_full((n_samples,), t_, dtype=torch.long)

            if t_ % interval == 0:
                xt = self.diffusion.p_sample(xt, t)
                frames.append(x0[0])

                if not create_video:
                    self.show_image(xt[0], f"{t_}")
                continue
            xt = self.diffusion.p_sample(xt, t)

        if create_video:
            self.make_video(frames)
    
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, lambda_: float = 0.01, t_: int = 100): #t_ is the final noised time stamp
        n_samples = x1.shape[0]
        t = torch.full((n_samples,), t_, device=self.device)
        xt = (1 - lambda_) * self.diffusion.q_sample(x1, t) + lambda_ * self.diffusion.q_sample(x2, t)
        return self._sample_x0(xt, t_)

    def interpolate_animate(self, x1: torch.Tensor, x2: torch.Tensor, n_frames: int = 100, t_: int = 10, create_video: bool = True):
        self.show_image(x1, "x1")
        self.show_image(x2, "x2")

        x1 = x1[None, :, :, :]
        x2 = x2[None, :, :, :]

        t = torch.full((1,), t_, device=self.device)
        x1t = self.diffusion.q_sample(x1, t)
        x2t = self.diffusion.q_sample(x2, t)

        frames = []
        for i in range(n_frames + 1):
            lambda_ = i / n_frames
            xt = (1 - lambda_) * x1t + lambda_ * x2t
            x0 = self._sample_x0(xt, t_)
            frames.append(x0[0])

            if not create_video:
                self.show_image(x0[0], f"{lambda_ :.2f}")
        
        if create_video:
            self.make_video(frames)

def main(animate=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion_model = torch.load("ddpm_model.pth", map_location=device)
    sampler = Sampler(diffusion_model, image_channels=1, image_size=32, device=device) # MNIST data has 1 channel

    with torch.no_grad():
        if sample_type == "image":
            if animate:
                sampler.sample_animation(n_frames=2)
            else:
                sampler.sample(n_samples=2)
        elif sample_type == "interpolate":
            dataset = MNISTDataset(image_size=32)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True)
            data = next(iter(data_loader)).to(device)
            
            if animate:
                sampler.interpolate_animate(data[0], data[1])
            else:
                interpolated_image = sampler.interpolate(data[0], data[1])
                sampler.show_image(interpolated_image, "interpolated image")
            
if __name__ == "__main__":
    main()
