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
        self.eps_model = diffusion.eps_model
        self.beta = diffusion.beta
        self.alpha = diffusion.alpha
        self.alpha_bar = diffusion.alpha_bar
        alpha_bar_tm1 = torch.cat([self.alpha_bar.new_ones((1,)), self.alpha_bar[:-1]])
        self.beta_tilde = self.beta * (1 - alpha_bar_tm1) / (1 - self.alpha_bar)
        self.mu_tilde_coef2 = (self.alpha ** 0.5) * (1 - alpha_bar_tm1) / (1 - self.alpha_bar)
        self.sigma2 = self.beta 
    
    def show_image(self, img, title=""):
        img = img.clip(0, 1)
        img = img.cpu().numpy()
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
    
    def sample_animation(self, n_frames: int = 1000, create_video: bool = True):
        xt = torch.randn([1, self.image_channels, self.image_size, self.image_size], device=self.device)
        interval = self.n_steps // n_frames
        frames = []

        for t_inv in range(self.n_steps):
            t_ = self.n_steps - t_inv - 1
            t = xt.new_full((1,), t_, dtype=torch.long)
            eps_theta = self.eps_model(xt, t)

            if t_ % interval == 0:
                x0 = self.p_x0(xt, t, eps_theta)
                frames.append(x0[0])

                if not create_video:
                    self.show_image(x0[0], f"{t_}")
            
            xt = sel.p_sample(xt, t, eps_theta)

        if create_video:
            self.make_video(frames)
    
    def interpolate():
        pass

    def interpolate_animate():
        pass

    def _sample_x0():
        pass

    def sample(self):
        pass 
    
    def p_sample(self):
        pass

    def p_x0(self):
        pass

def main():
    diffusion_model = torch.load("./checkpoints/")
    sampler = Sampler(diffusion_model, image_channels=3, image_size=32, device=device)

    with torch.no_grad():
        sampler.sample_animation()

        if False:
            data = next(iter(data_loader)).to(device)
            sampler.interpolate_animate(data[0], data[1])

if __name__ == "__main__":
    main()