import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    species_index = {s['name']: i for i, s in enumerate(species)}
    n_species = len(species)
    y0 = np.array([s['initial'] for s in species])
    
    def deriv(t, y):
        dydt = np.zeros(n_species)
        for reaction in network['reactions']:
            product = 1.0
            for species_name, stoich in reaction['reactants'].items():
                product *= y[species_index[species_name]] ** stoich
            rate = reaction['rate'] * product
            # Subtract reactants
            for species_name, stoich in reaction['reactants'].items():
                dydt[species_index[species_name]] -= stoich * rate
            # Add products
            for species_name, stoich in reaction['products'].items():
                dydt[species_index[species_name]] += stoich * rate
        return dydt
    
    sol = solve_ivp(deriv, [0, t_eval[-1]], y0, t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-8)
    return sol.y.T
