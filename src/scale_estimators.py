from scipy.stats import median_abs_deviation

def mad_scale(y, delta):
    y_obs = y[delta == 1]
    return median_abs_deviation(y_obs, nan_policy ='omit')