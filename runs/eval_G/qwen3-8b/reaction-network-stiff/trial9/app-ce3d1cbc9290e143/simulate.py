import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    n_species = len(species)
    initial_concentrations = [s['initial'] for s in species]

    species_names = [s['name'] for s in species]
    species_index = {name: i for i, name in enumerate(species_names)}

    reactions = network['reactions']

    def dydt(t, y):
        dy = np.zeros(n_species)
        for reaction in reactions:
            rate = reaction['rate']
            reactants = reaction['reactants']
            reactant_product = 1.0
            for species_name, stoich in reactants.items():
                idx = species_index[species_name]
                reactant_product *= y[idx] ** stoich
            reaction_rate = rate * reactant_product

            products = reaction['products']
            for species_name, stoich in products.items():
                idx = species_index[species_name]
                dy[idx] += stoich * reaction_rate
            for species_name, stoich in reactants.items():
                idx = species_index[species_name]
                dy[idx] -= stoich * reaction_rate
        return dy

    sol = solve_ivp(dydt, [0, network['t_end']], initial_concentrations, t_eval=t_eval, method='LSODA', rtol=1e-6, atol=1e-6)
    return sol.y.T