"""
Evaluation  module for robust nonparametric regression estimators.
Computes MISE and generates plots.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from slms import slms_estimator
from kernels import gaussian_kernel
from psi_functions import huber_psi

def compute_mise(true_func,estimator,x,y,delta,h,gamma=None,psi=None,kernel=None,imputation_kernel=None,n_rep=200):
    """
    Compute MISE over n_rep simulations.
    """
    R=50
    v_graid = np.linspace(0.1,0.9,R)
    mise = 0.0

    for _ in range(n_rep):
        x_sim, y_sim ,delta_sim = simulate_data(n=len(x))

        if estimator.__name__ == 'ilms_estimator':
            m_hat = [estimator(x_sim, y_sim, delta_sim,x0, h, gamma, psi, kernel, imputation_kernel) for x0 in v_graid]
        else:
            m_hat = [estimator(x_sim, y_sim, delta_sim,x0, h, gamma, psi, kernel) for x0 in v_graid]

            m_true = true_func(v_graid)
            ise = np.mean((m_true - m_hat)**2)
            mise += ise
        return mise / n_rep
    
def plot_comparison(x,y,delta,slms_estimates,ilms_estimates,nw_estimates,x_grid,true_func):
    """
     Plot observed, missing, true function, SLMS ,and ILMS.
    """
    plt.figure(figsize=(10,6))
    plt.scatter(x[delta==1],y[delta==1],color='blue',label='Observed ',alpha=0.6)
    plt.scatter(x[delta==0],y[delta==0],color='red',label='Missing ',alpha=0.6)
    plt.plot(x_grid,true_func(x_grid),'k-',label='True $m(x)',lw=2)
    plt.plot(x_grid,slms_estimates,'r--',label='SLMS ',lw=2)
    plt.plot(x_grid,ilms_estimates,'g-',label='ILMS ',lw=2)
    plt.plot(x_grid,nw_estimates, 'b:', label='Nadaraya-Watson', lw=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(' SLMS vs ILMS Estimation vs Nadaraya-Watson')
    plt.legend()
    #plt.savefig('results/plots/simulation_comparison.png',dpi=300,bbox_inches='tight')
    plt.show()

def robust_cross_validation(x,y,delta,h,psi,kernel,scale_estimator,v_func=None):
    """
    compute robust Cross-validation (RCV) for SLMS.
    RCV1,S(H) = sum δ_i * ρ_H( u_i / σ_n(x_i) ) * v(x_i)
    """
    if v_func is None:
        v_func = np.ones(len(x))
    else:
        v_func = v_func(x)
    
    rcv = 0.0
    for i in range(len(x)):
        if delta[i] == 0:
            continue # skip missing observations

        # Leave-one-out estimate at x[i]
        mask = np.ones(len(x), dtype=bool)
        mask[i] = False
        x_loo = x[mask]
        y_loo = y[mask]
        delta_loo = delta[mask]

        m_hat_loo = slms_estimator(
            x=x_loo,
            y=y_loo,
            delta=delta_loo,
            x0=x[i],h=h,
            psi=psi,
            kernel=kernel
        )

        u_i = y[i] - m_hat_loo
        sigma_hat = scale_estimator(y_loo[delta_loo==1])
        if sigma_hat == 0:
            sigma_hat = 1e-8  # Prevent division by zero

        # Use Huber's rho function
        rho_H = lambda u: np.where(np.abs(u) <= 1.345, 0.5 * u**2, 1.345 * (np.abs(u) - 0.5 * 1.345**2))
        rcv += delta[i] * rho_H(u_i / sigma_hat) * v_func[i]

    return rcv

def select_bandwidth(x,y,delta,h_grid,psi,kernel,scale_estimator,v_func=None):
    """
    Select optimal h by minimizing (RCV).
    """
    rcv_values = []
    for h in h_grid:
        rcv = robust_cross_validation(x,y,delta,h,psi,kernel,scale_estimator,v_func)
        rcv_values.append(rcv)
    
    optimal_h = h_grid[np.argmin(rcv_values)]
    return optimal_h, rcv_values



