import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    reactions = network['reactions']
    species_names = [s['name'] for s in species]
    species_dict = {name: idx for idx, name in enumerate(species_names)}
    n_species = len(species_names)

    y0 = np.array([s['initial'] for s in species])

    def dydt(t, y):
        dydt = np.zeros(n_species)
        for reaction in reactions:
            rate = reaction['rate']
            reactants = reaction['reactants']
            products = reaction['products']
            product = 1.0
            for species_name, stoich in reactants.items():
                species_index = species_dict[species_name]
                product *= y[species_index] ** stoich
            rate *= product
            for species_name in species_names:
                reactant_stoich = reactants.get(species_name, 0)
                product_stoich = products.get(species_name, 0)
                delta = product_stoich - reactant_stoich
                dydt[species_dict[species_name]] += delta * rate
        return dydt

    sol = solve_ivp(dydt, [0, network['t_end']], y0, t_eval=t_eval, method='BDF', rtol=1e-6, atol=1e-6)
    return sol.y.T
