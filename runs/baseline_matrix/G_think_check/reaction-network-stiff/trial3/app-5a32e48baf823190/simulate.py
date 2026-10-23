import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    y0 = np.array([s['initial'] for s in species])
    species_indices = {s['name']: idx for idx, s in enumerate(species)}

    def dydt(t, y):
        dy = np.zeros_like(y)
        for reaction in network['reactions']:
            rate = reaction['rate']
            # Compute rate
            for reactant, stoich in reaction['reactants'].items():
                rate *= y[species_indices[reactant]] ** stoich
            # Update dy for products and reactants
            for product, stoich in reaction['products'].items():
                dy[species_indices[product]] += stoich * rate
            for reactant, stoich in reaction['reactants'].items():
                dy[species_indices[reactant]] -= stoich * rate
        return dy

    sol = solve_ivp(dydt, (0.0, network['t_end']), y0, t_eval=t_eval, method='Radau', rtol=1e-8, atol=1e-8)
    return sol.y.T
