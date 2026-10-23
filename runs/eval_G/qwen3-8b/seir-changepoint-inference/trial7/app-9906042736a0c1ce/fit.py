import numpy as np
import scipy.optimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    def simulate_model(R, initial_S, initial_E, initial_I, initial_R, days):
        S = initial_S
        E = initial_E
        I = initial_I
        R_ = initial_R
        predicted_cases = []
        for day in days:
            new_infections = R * gamma * I * S / population
            new_infectious = sigma * E
            new_removed = gamma * I
            S -= new_infections
            E += new_infections - new_infectious
            I += new_infectious - new_removed
            R_ += new_removed
            predicted_cases.append(new_infectious)
        return predicted_cases, S, E, I, R_ 

    best_error = float('inf')
    best_change_day = None
    best_R0 = None
    best_R_post = None

    for change_day_candidate in [None] + list(range(1, 71)):
        if change_day_candidate is None:
            def error_function_R(R):
                days = list(range(1, 71))
                predicted_cases, _, _, _, _ = simulate_model(R, population - initial_infectious, 0, initial_infectious, 0, days)
                observed = cases
                # Poisson log-likelihood
                log_likelihood = 0.0
                for pred, obs in zip(predicted_cases, observed):
                    if pred < 0:
                        pred = 0
                    log_likelihood += -pred + obs * np.log(pred) - np.log(1 + np.exp(-pred))
                return -log_likelihood
            try:
                result = scipy.optimize.minimize(error_function_R, x0=1.0, bounds=[(0.1, 10.0)])
                best_R = result.x[0]
                total_error = result.fun
            except:
                best_R = 1.0
                total_error = float('inf')
        else:
            days_part1 = list(range(1, change_day_candidate))
            observed_cases_part1 = cases[0:change_day_candidate - 1]
            
            def error_function_R0(R):
                predicted_cases, _, _, _, _ = simulate_model(R, population - initial_infectious, 0, initial_infectious, 0, days_part1)
                observed = observed_cases_part1
                log_likelihood = 0.0
                for pred, obs in zip(predicted_cases, observed):
                    if pred < 0:
                        pred = 0
                    log_likelihood += -pred + obs * np.log(pred) - np.log(1 + np.exp(-pred))
                return -log_likelihood
            try:
                result_R0 = scipy.optimize.minimize(error_function_R0, x0=1.0, bounds=[(0.1, 10.0)])
                best_R0_candidate = result_R0.x[0]
            except:
                best_R0_candidate = 1.0
            predicted_cases_part1, final_S, final_E, final_I, final_R = simulate_model(best_R0_candidate, population - initial_infectious, 0, initial_infectious, 0, days_part1)
            
            days_part2 = list(range(change_day_candidate, 71))
            observed_cases_part2 = cases[change_day_candidate - 1:70]
            
            def error_function_R_post(R):
                predicted_cases, _, _, _, _ = simulate_model(R, final_S, final_E, final_I, final_R, days_part2)
                observed = observed_cases_part2
                log_likelihood = 0.0
                for pred, obs in zip(predicted_cases, observed):
                    if pred < 0:
                        pred = 0
                    log_likelihood += -pred + obs * np.log(pred) - np.log(1 + np.exp(-pred))
                return -log_likelihood
            try:
                result_R_post = scipy.optimize.minimize(error_function_R_post, x0=1.0, bounds=[(0.1, 10.0)])
                best_R_post_candidate = result_R_post.x[0]
            except:
                best_R_post_candidate = 1.0
            total_error = result_R0.fun + result_R_post.fun
        
        if total_error < best_error:
            best_error = total_error
            best_change_day = change_day_candidate
            if change_day_candidate is None:
                best_R0 = best_R
                best_R_post = best_R
            else:
                best_R0 = best_R0_candidate
                best_R_post = best_R_post_candidate

    if best_change_day is None:
        R_forecast = best_R0
        days_forecast_full = list(range(1, 71))
        predicted_cases_full, final_S, final_E, final_I, final_R = simulate_model(R_forecast, population - initial_infectious, 0, initial_infectious, 0, days_forecast_full)
        forecast_days = list(range(71, 70 + horizon + 1))
        predicted_forecast, _, _, _, _ = simulate_model(R_forecast, final_S, final_E, final_I, final_R, forecast_days)
        forecast = predicted_forecast
    else:
        days_part1 = list(range(1, best_change_day))
        predicted_cases_part1, final_S_part1, final_E_part1, final_I_part1, final_R_part1 = simulate_model(best_R0, population - initial_infectious, 0, initial_infectious, 0, days_part1)
        days_part2 = list(range(best_change_day, 71))
        predicted_cases_part2, final_S_part2, final_E_part2, final_I_part2, final_R_part2 = simulate_model(best_R_post, final_S_part1, final_E_part1, final_I_part1, final_R_part1, days_part2)
        forecast_days = list(range(71, 70 + horizon + 1))
        predicted_forecast, _, _, _, _ = simulate_model(best_R_post, final_S_part2, final_E_part2, final_I_part2, final_R_part2, forecast_days)
        forecast = predicted_forecast

    return {
        "R0": best_R0,
        "R_post": best_R_post,
        "change_day": best_change_day,
        "forecast": forecast
    }