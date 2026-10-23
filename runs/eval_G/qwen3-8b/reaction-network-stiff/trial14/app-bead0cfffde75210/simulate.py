import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    initial = np.array([species['initial'] for species in network['species']])
    species_index = {species['name']: idx for idx, species in enumerate(network['species'])}
    species_names = [species['name'] for species in network['species']]

    def dydt(t, y):
        dy = np.zeros_like(y)
        for reaction in network['reactions']:
            rate = reaction['rate']
            reactants = reaction['reactants']
            products = reaction['products']
            product = 1.0
            for species_name, stoich in reactants.items():
                idx = species_index[species_name]
                product *= y[idx] ** stoich
            reaction_rate = rate * product
            for species_name in species_names:
                stoich_products = products.get(species_name, 0)
                stoich_reactants = reactants.get(species_name, 0)
                dy[species_index[species_name]] += (stoich_products - stoich_reactants) * reaction_rate
        return dy

    sol = solve_ivp(dydt, [0, network['t_end']], initial, t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-10)
    return sol.y.T
