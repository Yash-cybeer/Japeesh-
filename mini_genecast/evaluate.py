# mini_gencast/evaluate.py
import numpy as np

def rmse(pred, truth):
    return np.sqrt(np.mean((pred - truth)**2))

def ensemble_mean(ens):
    # ens: (N_ens, H, W)
    return np.mean(ens, axis=0)

def ensemble_spread(ens):
    # ens: (N_ens, H, W)
    return np.std(ens, axis=0)
