import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    species_index = {s['name']: i for i, s in enumerate(species)}
    y0 = np.array([s['initial'] for s in species])
    t_end = network['t_end']

    def dydt(t, y):
        derivatives = np.zeros_like(y)
        for reaction in network['reactions']:
            reactants = reaction['reactants']
            products = reaction['products']
            rate = reaction['rate']
            r = rate
            for species_name, stoich in reactants.items():
                r *= y[species_index[species_name]]
            for species_name, stoich in products.items():
                derivatives[species_index[species_name]] += stoich * r
            for species_name, stoich in reactants.items():
                derivatives[species_index[species_name]] -= stoich * r
        return derivatives

    sol = solve_ivp(dydt, [0.0, t_end], y0, t_eval=t_eval, method='BDF')
    return sol.y.T
