# mini_gencast/diffusion.py
import torch

def linear_beta_schedule(T=20, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

class Diffusion:
    def __init__(self, model, T=20, device='cpu'):
        self.model = model
        self.T = T
        self.device = device
        self.beta = linear_beta_schedule(T).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, z0, t):
        B = z0.size(0)
        noise = torch.randn_like(z0).to(self.device)
        a_bar = self.alpha_bar[t].view(B,1,1,1)
        noisy = torch.sqrt(a_bar) * z0 + torch.sqrt(1. - a_bar) * noise
        return noisy, noise
