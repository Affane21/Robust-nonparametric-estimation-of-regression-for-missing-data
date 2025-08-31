import numpy as np 
def missing_mechanism(x):
    p_x = 0.3 + 0.5 * np.sin(5 * (x + 0.2))**2 
    delta = np.random.rand(len(x)) < p_x
    return delta.astype(int)
