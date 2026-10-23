import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    reactions = network['reactions']
    t_end = network['t_end']
    n_species = len(species)

    species_index = {s['name']: i for i, s in enumerate(species)}

    y0 = np.array([s['initial'] for s in species])

    def dydt(t, y):
        dydt = np.zeros(n_species)
        for reaction in reactions:
            k = reaction['rate']
            reactants = reaction['reactants']
            products = reaction['products']
            rate = k
            for species_name, stoich in reactants.items():
                rate *= y[species_index[species_name]] ** stoich
            for species_name, stoich in reactants.items():
                dydt[species_index[species_name]] -= stoich * rate
            for species_name, stoich in products.items():
                dydt[species_index[species_name]] += stoich * rate
        return dydt

    sol = solve_ivp(dydt, [0.0, t_end], y0, t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-8)
    return sol.y.T
