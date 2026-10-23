import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    reactions = network['reactions']
    t_end = network['t_end']

    species_dict = {s['name']: i for i, s in enumerate(species)}
    y0 = np.array([s['initial'] for s in species])

    def dydt(t, y):
        dydt = np.zeros_like(y)
        for reaction in reactions:
            rate = reaction['rate']
            # Compute reactant terms
            reactant_terms = 1.0
            for name, stoich in reaction['reactants'].items():
                idx = species_dict[name]
                reactant_terms *= y[idx] ** stoich
            rate *= reactant_terms
            # Update dydt for reactants and products
            for name, stoich in reaction['reactants'].items():
                idx = species_dict[name]
                dydt[idx] -= stoich * rate
            for name, stoich in reaction['products'].items():
                idx = species_dict[name]
                dydt[idx] += stoich * rate
        return dydt

    sol = solve_ivp(dydt, [0.0, t_end], y0, t_eval=t_eval, method='Radau', rtol=1e-10, atol=1e-12)
    return sol.y.T