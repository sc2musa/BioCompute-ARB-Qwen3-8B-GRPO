import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    species_index = {s['name']: idx for idx, s in enumerate(species)}
    y0 = [s['initial'] for s in species]
    reactions = network['reactions']

    def dydt(t, y):
        dy = np.zeros_like(y)
        for reaction in reactions:
            rate = reaction['rate']
            for reactant, stoich in reaction['reactants'].items():
                idx = species_index[reactant]
                rate *= y[idx] ** stoich
            for reactant, stoich in reaction['reactants'].items():
                idx = species_index[reactant]
                dy[idx] -= stoich * rate
            for product, stoich in reaction['products'].items():
                idx = species_index[product]
                dy[idx] += stoich * rate
        return dy

    sol = solve_ivp(dydt, [0, network['t_end']], y0, t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-9)
    return sol.y.T
