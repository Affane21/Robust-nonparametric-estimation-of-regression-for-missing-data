"""
Imputed Local M-Smoother (ILMS) for robust nonparametric regression with missing data.
Based on Affane (2024) and Boente et al. (2009).
"""
import numpy as np
from slms import slms_estimator

def ilms_estimator(x, y,delta,x0,h,gamma,psi, kernel,imputation_kernel, max_iter=100, tol=1e-6):
    """
    Estimate m(x0) using the Imputed Local M-Smoother (ILMS).
    parameters:
    - x, y, delta: data
    - x0: point to estimate
    - h: bandwidth for SLMS (imputation)
    - gamma: bandwidth for final smoothing
    -psi: influence function
    - kernel: kernel for final smoothing
    - imputation_kernel: kernel for SLMS inputation
    - max_iter, tol: convergence parameters

    returns:
    - m_hat: ILMS estimated at x0 
    
    """
    # Step 1: Impute missing values using SLMS
    y_imputed = y.copy()
    for i in range(len(x)):
        if delta[i] == 0:
            y_imputed[i] = slms_estimator(
                x=x,
                y=y,
                delta=delta,
                x0=x[i],
                h=h,
                psi=psi,
                kernel=imputation_kernel,
                max_iter=max_iter,
                tol=tol
            )
    # Step 2: Apply SLMS on the imputed data
    m_hat = slms_estimator(
        x=x,
        y=y_imputed,
        delta=np.ones(len(x)),  # All data is now "observed"
        x0=x0,
        h=gamma,
        psi=psi,
        kernel=kernel,
        max_iter=max_iter,
        tol=tol
    )

    return m_hat