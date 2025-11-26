# mini_gencast/sample.py
import torch
import numpy as np
from mini_gencast.model import Denoiser
import meshzoo

def simple_sample(model, diff, Xt_1, steps=None, device='cpu'):
    model.eval()
    T = diff.T
    if steps is None:
        steps = T
    z = torch.randn_like(Xt_1).to(device)
    with torch.no_grad():
        for t in reversed(range(steps)):
            pred = model(z, Xt_1)
            z = z - diff.beta[t] * pred
    Xt = Xt_1 + z
    return Xt
