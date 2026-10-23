import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    y0 = np.array([s['initial'] for s in species])
    species_index = {s['name']: i for i, s in enumerate(species)}

    def dydt(t, y):
        dy = np.zeros_like(y)
        for reaction in network['reactions']:
            rate = reaction['rate']
            reactants = reaction['reactants']
            products = reaction['products']
            reactant_conc = []
            for species_name, stoich in reactants.items():
                if species_name in species_index:
                    reactant_conc.append(y[species_index[species_name]] ** stoich)
            if not reactants:
                rate_val = rate
            else:
                rate_val = rate * np.prod(reactant_conc)
            for species_name, stoich in products.items():
                if species_name in species_index:
                    idx = species_index[species_name]
                    dy[idx] += stoich * rate_val
            for species_name, stoich in reactants.items():
                if species_name in species_index:
                    idx = species_index[species_name]
                    dy[idx] -= stoich * rate_val
        return dy

    t_end = network['t_end']
    sol = solve_ivp(dydt, [0.0, t_end], y0, t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-8)
    return sol.y.T