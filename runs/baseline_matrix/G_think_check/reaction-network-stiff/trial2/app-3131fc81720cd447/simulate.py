import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    species_index = {s['name']: i for i, s in enumerate(species)}
    y0 = np.array([s['initial'] for s in species])
    t_end = network['t_end']
    
    def dydt(t, y):
        dy = np.zeros_like(y)
        for reaction in network['reactions']:
            rate = reaction['rate']
            reactants = reaction['reactants']
            products = reaction['products']
            # Compute the rate value
            rate_value = rate
            for species_name, stoich in reactants.items():
                idx = species_index[species_name]
                rate_value *= y[idx] ** stoich
            # Update dy for products and reactants
            for species_name, stoich in products.items():
                idx = species_index[species_name]
                dy[idx] += stoich * rate_value
            for species_name, stoich in reactants.items():
                idx = species_index[species_name]
                dy[idx] -= stoich * rate_value
        return dy
    
    sol = solve_ivp(dydt, [0.0, t_end], y0, t_eval=t_eval, method='LSODA', rtol=1e-6, atol=1e-6)
    return sol.y.T
