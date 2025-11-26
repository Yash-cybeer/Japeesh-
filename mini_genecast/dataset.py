# mini_gencast/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

class SyntheticWeatherDataset(Dataset):
    """
    Synthetic dataset to emulate ERA5-like sequences (T2M, U10, V10).
    Each item returns:
      Xt_1: previous state (3,H,W)
      residual: Xt - Xt_1 (3,H,W)
    """
    def __init__(self, length=1000, H=64, W=128):
        self.length = length
        self.H = H
        self.W = W
        x = np.linspace(0, 3*np.pi, W)
        y = np.linspace(0, 3*np.pi, H)
        self.X, self.Y = np.meshgrid(x, y)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        X, Y = self.X, self.Y
        T2M  = np.sin(X + idx*0.05) * np.cos(Y + idx*0.03)
        U10  = np.cos(X*0.7  + idx*0.1)
        V10  = np.sin(Y*0.9  + idx*0.1)
        Xt_1 = np.stack([T2M, U10, V10]).astype(np.float32)

        T2M2 = np.sin(X + (idx+1)*0.05) * np.cos(Y + (idx+1)*0.03)
        U102 = np.cos(X*0.7  + (idx+1)*0.1)
        V102 = np.sin(Y*0.9  + (idx+1)*0.1)
        Xt = np.stack([T2M2, U102, V102]).astype(np.float32)

        residual = (Xt - Xt_1).astype(np.float32)

        return torch.from_numpy(Xt_1), torch.from_numpy(residual)

class ERA5SequenceDataset(Dataset):
    """
    Simple wrapper: expects preprocessed NetCDFs with variables stacked.
    """
    def __init__(self, nc_files, variables=['t2m','u10','v10']):
        ds = xr.open_mfdataset(nc_files, combine='by_coords')
        self.vars = variables
        self.ds = ds.sel(time=ds.time)
        self.times = self.ds.time.values
        self.n = len(self.times)

    def __len__(self):
        return max(0, self.n - 1)

    def __getitem__(self, idx):
        t0 = idx
        t1 = idx + 1
        Xt_1 = np.stack([self.ds[v].isel(time=t0).values for v in self.vars], axis=0).astype(np.float32)
        Xt   = np.stack([self.ds[v].isel(time=t1).values for v in self.vars], axis=0).astype(np.float32)
        residual = Xt - Xt_1
        return torch.from_numpy(Xt_1), torch.from_numpy(residual)
