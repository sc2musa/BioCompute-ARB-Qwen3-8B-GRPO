import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    species_dict = {s['name']: i for i, s in enumerate(species)}
    n_species = len(species)
    y0 = np.array([s['initial'] for s in species])
    t_end = network['t_end']

    def dydt(t, y):
        derivatives = np.zeros(n_species)
        for reaction in network['reactions']:
            rate = reaction['rate']
            for reactant, stoich in reaction['reactants'].items():
                rate *= y[species_dict[reactant]] ** stoich
            for species, stoich in reaction['reactants'].items():
                idx = species_dict[species]
                derivatives[idx] -= rate * stoich
            for species, stoich in reaction['products'].items():
                idx = species_dict[species]
                derivatives[idx] += rate * stoich
        return derivatives

    sol = solve_ivp(dydt, (0.0, t_end), y0, t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-8)
    return sol.y.T