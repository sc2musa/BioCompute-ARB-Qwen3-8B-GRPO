import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    n_species = len(species)
    species_to_index = {s['name']: i for i, s in enumerate(species)}
    y0 = [s['initial'] for s in species]

    def dydt(t, y):
        derivatives = np.zeros(n_species)
        for reaction in network['reactions']:
            rate = reaction['rate']
            reactants = reaction['reactants']
            products = reaction['products']
            product = 1.0
            for species_name, stoich in reactants.items():
                index = species_to_index[species_name]
                product *= y[index] ** stoich
            rate *= product
            for species_name, stoich in products.items():
                index = species_to_index[species_name]
                derivatives[index] += stoich * rate
            for species_name, stoich in reactants.items():
                index = species_to_index[species_name]
                derivatives[index] -= stoich * rate
        return derivatives

    t_span = (0.0, network['t_end'])
    sol = solve_ivp(dydt, t_span, y0, t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-8)
    return sol.y.T
