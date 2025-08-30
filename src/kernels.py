import numpy as np
def gaussian_kernel(u):
    return np.exp(-0.5 * u**2)

def epanechnikov_kernel(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)