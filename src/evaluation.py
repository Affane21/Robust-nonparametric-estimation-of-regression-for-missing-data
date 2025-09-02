"""
Evaluation  module for robust nonparametric regression estimators.
Computes MISE and generates plots.
"""
import numpy as np
import matplotlib.pyplot as plt

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
        