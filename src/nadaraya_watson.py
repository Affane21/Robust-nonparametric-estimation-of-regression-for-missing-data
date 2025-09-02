"""
Classical Nadaraya-Watson estimator for nonparametric regression.
"""

import numpy as np

def nadaraya_watson(x, y, delta, x0, h, kernel):
    """
    Estimate m(x0) using the classical Nadaraya-Watson estimator.
    """
    numerator = 0.0
    denominator = 0.0
    
    for i in range(len(x)):
        if delta[i] == 0:
            continue  # Skip missing responses
        
        u = (x[i] - x0) / h
        K = kernel(u)
        numerator += K * y[i]
        denominator += K
    
    if denominator == 0:
        return np.nan
    
    return numerator / denominator