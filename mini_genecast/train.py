# mini_gencast/train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from mini_gencast.dataset import SyntheticWeatherDataset
from mini_gencast.model import Denoiser
from mini_gencast.diffusion import Diffusion

def train_simple(save_path='model.pth', epochs=3, batch_size=4, device='cpu'):
    ds = SyntheticWeatherDataset(length=500, H=64, W=128)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    import meshzoo
    pts, cells = meshzoo.icosa_sphere(2)
    mesh_nodes = pts.shape[0]

    model = Denoiser(mesh_nodes).to(device)
    diff = Diffusion(model, T=20, device=device)
    opt = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        running = 0.0
        for i, (Xt_1, residual) in enumerate(dl):
            Xt_1 = Xt_1.to(device)
            residual = residual.to(device)
            B = residual.size(0)
            t = torch.randint(0, diff.T, (B,), device=device)
            noisy_res, noise = diff.q_sample(residual, t)
            pred = model(noisy_res, Xt_1)
            loss = ((pred - noise)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
            if (i+1) % 50 == 0:
                print(f"Epoch {epoch+1} step {i+1} loss {running/50:.6f}")
                running = 0.0
    torch.save(model.state_dict(), save_path)
    print("Saved", save_path)
    return model, diff
