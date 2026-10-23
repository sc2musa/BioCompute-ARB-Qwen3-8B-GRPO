import numpy as np
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Implement the estimator here
    # This is a placeholder implementation
    return {
        "R0": 2.0,
        "R_post": 2.0,
        "change_day": None,
        "forecast": [100.0] * horizon
    }