import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(39) 
def data_simulation(n=100,sigma=0.5,eta=0.1):
    """
    Simulate data for regression with missing responses.
    
    Parameters:
    n : number of observations.
    sigma : standard deviation of the noise.
    eta : contamination proportion.
    
    Returns:
    x : array of covariates.
    y : array of responses with noise.
    y_true : true regression function values.
delta : indicator of observed responses (1 = observed, 0 = missing).
"""
    x = np.random.uniform(0,1,n)

    clean_noise = np.random.normal(0,sigma,n)
    contaminated_noise = np.random.normal(0,0.25*sigma,n)
    epsilon = (1-eta)*clean_noise + eta*contaminated_noise

    y_true = 0.25 * np.pi * np.sin(np.pi * x)
    y = y_true + epsilon
    return x, y

def simulate_once(n=100, eta=0.1, sigma=0.5):
    """Simulate one dataset with MAR and contamination."""
    np.random.seed()  # Seed متغير لكل محاكاة
    
    x = np.random.uniform(0, 1, n)
    
    # Contamination
    clean_noise = np.random.normal(0, sigma, n)
    contaminated_noise = np.random.normal(0, 5*sigma, n)
    epsilon = np.where(np.random.rand(n) < eta, contaminated_noise, clean_noise)
    
    y = 0.25 * np.sin(np.pi * x) + epsilon
    
    # MAR mechanism
    p_x = 0.3 + 0.5 * np.sin(5 * (x + 0.2))**2
    delta = np.random.rand(n) < p_x
    
    return x, y, delta