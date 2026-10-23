import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    species_index = {s['name']: idx for idx, s in enumerate(species)}
    n_species = len(species)
    y0 = np.array([s['initial'] for s in species])

    def dydt(t, y):
        dydt = np.zeros(n_species)
        for reaction in network['reactions']:
            rate = reaction['rate']
            rate_value = rate
            for name, stoich in reaction['reactants'].items():
                idx = species_index[name]
                rate_value *= y[idx] ** stoich
            for name, stoich in reaction['reactants'].items():
                idx = species_index[name]
                dydt[idx] -= stoich * rate_value
            for name, stoich in reaction['products'].items():
                idx = species_index[name]
                dydt[idx] += stoich * rate_value
        return dydt

    sol = solve_ivp(dydt, [0.0, network['t_end']], y0, t_eval=t_eval, method='BDF', rtol=1e-6, atol=1e-8)
    return sol.y.T
