# mini_gencast/model.py
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_ch=6, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class GridToMesh(nn.Module):
    def __init__(self, hidden, mesh_nodes):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)
        self.mesh_nodes = mesh_nodes
    def forward(self, x):
        B, C, H, W = x.shape
        pooled = x.mean(dim=[2,3])
        pooled = self.lin(pooled)
        return pooled.unsqueeze(1).repeat(1, self.mesh_nodes, 1)

class MeshProcessor(nn.Module):
    def __init__(self, hidden=64, layers=3, heads=4):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, batch_first=True)
        self.tf = nn.TransformerEncoder(enc, num_layers=layers)
    def forward(self, x):
        return self.tf(x)

class MeshToGrid(nn.Module):
    def __init__(self, hidden=64, shape=(64,128)):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)
        self.H, self.W = shape
    def forward(self, x):
        pooled = x.mean(1)
        pooled = self.lin(pooled)
        return pooled[:,:,None,None].repeat(1,1,self.H,self.W)

class Decoder(nn.Module):
    def __init__(self, hidden=64, out_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, out_ch, 3, padding=1)
        )
    def forward(self, x):
        return self.net(x)

class Denoiser(nn.Module):
    def __init__(self, mesh_nodes, hidden=64):
        super().__init__()
        self.encoder = Encoder(in_ch=6, hidden=hidden)
        self.g2m = GridToMesh(hidden, mesh_nodes)
        self.proc = MeshProcessor(hidden)
        self.m2g = MeshToGrid(hidden)
        self.decoder = Decoder(hidden, out_ch=3)
    def forward(self, noisy_res, Xt_1):
        x = torch.cat([noisy_res, Xt_1], dim=1)
        e = self.encoder(x)
        m = self.g2m(e)
        m = self.proc(m)
        g = self.m2g(m)
        out = self.decoder(g)
        return out
