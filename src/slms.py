"""
Simplified local M-smoother (SLMS) for robust nonparametric regression with missing data.
Based on affane (2024) and Boente et al. (2009).
"""
import numpy as np
from scipy,stats import median_abs_deviation

def slms_estimator(x, y, delta, x0, h, k=1.345, tol=1e-6, max_iter=100):
    """
    Estimate m(x0) using the Simplified Local M-Smoother (SLMS).
    
    Parameters:
    x : array of covariates.
    y : array of Response.
    delta : indicator of observed responses (1 = observed, 0 = missing).
    x0 : Point at which to estimate m(x0).
    h : Bandwidth.
    psi : influence function (e.g.,Huber).
    kernel : kernel function (e.g., Gaussian).
    tol : convergence tolerance.
    max_iter : Maximum number of iterations.
        
    Returns:
    m_hat : robust estimate of m(x0).
    """
    # step 1: Compute Kernel weights (3.3)
    numerator = kernel((x - x0) / h) * delta
    denominator = np.sum(numerator)

    if denominator == 0:
        return np.nan  # No observed data in the neighborhood
    
    weights = numerator / denominator

    # step 2: Estimate scale (robust)
    y_obs = y[delta == 1]
    if len(y_obs) == 0:
        return np.nan
    sigma_hat = median_abs_deviation(y_obs, nan_policy='omit')

    # step 3: Initial estimate (weighted average)
    m_hat = np.average(y[delta == 1], weights=weights[delta == 1])

    # step 4 : Iterative reweighting (eq 3.6)
    for iteration in range(max_iter):
        residuals = y - m_hat
        scaled_residuals = residuals / (sigma_hat + 1e-8)
        psi_values = psi(scaled_residuals)

        # Compute u_weights = psi(u) / u if u != 0 else psi'(0)
        with np.errstate(divide='ignore', invalid='ignore'):
            u_weights = np.where(
                np.abs(scaled_residuals) > 1e-8,
                psi_vals / scaled_residuals,
                1.0 # psi'(0) for Huber/Bisquare
            )
        # final weights: kernel * robustness * missing indicator
        final_weights = weights * u_weights * delta

        if np.sum(final_weights) == 0:
            break  # Avoid division by zero

        m_hat_new = np.average(y, weights=final_weights)

        if abs(m_hat_new - m_hat) < tol:
            m_hat = n_hat_new
            break

        m_hat = m_hat_new
    
    return m_hat 