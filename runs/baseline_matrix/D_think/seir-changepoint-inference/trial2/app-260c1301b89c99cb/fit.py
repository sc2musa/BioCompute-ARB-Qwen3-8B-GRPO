import numpy as np
import scipy.optimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    def simulate(R0, R_post, change_day):
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        simulated_cases = []
        for day in range(1, 71):
            if change_day is not None and day >= change_day:
                beta = R_post * gamma
            else:
                beta = R0 * gamma
            new_cases = sigma * E
            delta_S = -beta * S * I / population
            delta_E = beta * S * I / population - sigma * E
            delta_I = sigma * E - gamma * I
            delta_R = gamma * I
            S += delta_S
            E += delta_E
            I += delta_I
            R += delta_R
            simulated_cases.append(new_cases)
        return simulated_cases

    # No intervention case
    def loss_no_intervention(R0):
        simulated = simulate(R0, R0, None)
        return np.sum((np.array(simulated) - np.array(cases)) ** 2)
    result_no_intervention = scipy.optimize.minimize(loss_no_intervention, x0=2.0, bounds=[(0.0, 100.0)])
    best_R0_no_intervention = result_no_intervention.x[0]
    best_loss_no_intervention = loss_no_intervention(best_R0_no_intervention)
    best_params_no_intervention = (best_R0_no_intervention, best_R0_no_intervention, None, best_loss_no_intervention)

    # Intervention cases
    best_params = best_params_no_intervention
    best_loss = best_loss_no_intervention

    for change_day in range(1, 71):
        def loss_intervention(params):
            R0, R_post = params
            if R_post > R0:
                R_post = R0
            simulated = simulate(R0, R_post, change_day)
            return np.sum((np.array(simulated) - np.array(cases)) ** 2)
        result_intervention = scipy.optimize.minimize(loss_intervention, x0=[2.0, 1.0], bounds=[(0.0, 100.0), (0.0, 100.0)])
        best_R0_intervention = result_intervention.x[0]
        best_R_post_intervention = result_intervention.x[1]
        best_loss_intervention = loss_intervention([best_R0_intervention, best_R_post_intervention])
        if best_loss_intervention < best_loss:
            best_params = (best_R0_intervention, best_R_post_intervention, change_day, best_loss_intervention)
            best_loss = best_loss_intervention

    R0, R_post, change_day, _ = best_params

    # Forecast
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    forecast = []
    for _ in range(horizon):
        if change_day is not None and (70 + 1) >= change_day:
            beta = R_post * gamma
        else:
            beta = R0 * gamma
        new_cases = sigma * E
        delta_S = -beta * S * I / population
        delta_E = beta * S * I / population - sigma * E
        delta_I = sigma * E - gamma * I
        delta_R = gamma * I
        S += delta_S
        E += delta_E
        I += delta_I
        R += delta_R
        forecast.append(new_cases)
    return {
        "R0": float(R0),
        "R_post": float(R_post),
        "change_day": float(change_day) if change_day is not None else None,
        "forecast": forecast
    }
