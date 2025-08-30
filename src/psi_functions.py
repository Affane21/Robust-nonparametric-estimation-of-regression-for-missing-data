import numpy as np
def huber_psi(u,k=1.345):
    return np.where(np.abs(u) <= k, u, k * np.sign(u))

def bisquare_psi(u,k=4.685):
    mask = np.abs(u) <= k
    psi_vals = np.zeros_like(u)
    psi_vals[mask] = u[mask] * (1 - (u[mask]/k)**2)**2
    return psi_vals