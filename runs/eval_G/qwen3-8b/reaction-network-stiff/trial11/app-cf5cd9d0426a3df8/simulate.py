import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species_list = network['species']
    reactions = network['reactions']
    species_names = [s['name'] for s in species_list]
    species_index = {name: idx for idx, name in enumerate(species_names)}
    n_species = len(species_list)
    initial_conditions = np.array([s['initial'] for s in species_list])

    def dydt(t, y):
        dy = np.zeros(n_species)
        for reaction in reactions:
            reactants = reaction['reactants']
            products = reaction['products']
            rate = reaction['rate']
            product = 1.0
            for species, stoich in reactants.items():
                idx = species_index[species]
                product *= y[idx] ** stoich
            reaction_rate = rate * product
            for species, stoich in products.items():
                idx = species_index[species]
                dy[idx] += stoich * reaction_rate
            for species, stoich in reactants.items():
                idx = species_index[species]
                dy[idx] -= stoich * reaction_rate
        return dy

    t_span = (0.0, t_eval[-1])
    sol = solve_ivp(dydt, t_span, initial_conditions, t_eval=t_eval, method='Radau', rtol=1e-8, atol=1e-8)
    return sol.y.T
