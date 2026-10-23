import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    species_index = {s['name']: i for i, s in enumerate(species)}
    y0 = [s['initial'] for s in species]
    t_end = network['t_end']

    def ode_func(t, y):
        dydt = np.zeros(len(y))
        for reaction in network['reactions']:
            rate = reaction['rate']
            # Compute the rate
            for species_name, stoich in reaction['reactants'].items():
                rate *= y[species_index[species_name]] ** stoich
            # Compute the net change for each species
            for species_name, idx in species_index.items():
                net_change = reaction['products'].get(species_name, 0) - reaction['reactants'].get(species_name, 0)
                dydt[idx] += net_change * rate
        return dydt

    sol = solve_ivp(ode_func, [0, t_end], y0, t_eval=t_eval, method='LSODA')
    return sol.y.T
