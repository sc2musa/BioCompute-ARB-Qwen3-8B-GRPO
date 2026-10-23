import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    n_species = len(species)
    species_to_idx = {s['name']: i for i, s in enumerate(species)}
    y0 = np.array([s['initial'] for s in species])

    def ode_func(t, y):
        dydt = np.zeros_like(y)
        for reaction in network['reactions']:
            rate = reaction['rate']
            # Compute the product of [species]^stoich for reactants
            for species_name, stoich in reaction['reactants'].items():
                rate *= y[species_to_idx[species_name]] ** stoich
            # Update dydt for reactants and products
            for species_name, stoich in reaction['reactants'].items():
                dydt[species_to_idx[species_name]] -= stoich * rate
            for species_name, stoich in reaction['products'].items():
                dydt[species_to_idx[species_name]] += stoich * rate
        return dydt

    sol = solve_ivp(ode_func, (0, network['t_end']), y0, t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-10)
    return sol.y.T