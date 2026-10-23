import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    initial_concentrations = [spec['initial'] for spec in species]
    species_index_map = {spec['name']: idx for idx, spec in enumerate(species)}

    def dydt(t, y):
        dydt = np.zeros_like(y)
        for reaction in network['reactions']:
            rate = reaction['rate']
            reactants = reaction['reactants']
            products = reaction['products']
            reactant_product = 1.0
            for species_name, stoich in reactants.items():
                idx = species_index_map[species_name]
                reactant_product *= y[idx] ** stoich
            reaction_rate = rate * reactant_product
            for spec in species:
                species_name = spec['name']
                idx = species_index_map[species_name]
                stoich_react = reactants.get(species_name, 0)
                stoich_prod = products.get(species_name, 0)
                dydt[idx] += (stoich_prod - stoich_react) * reaction_rate
        return dydt

    t_span = (0.0, network['t_end'])
    sol = solve_ivp(dydt, t_span, initial_concentrations, method='LSODA', t_eval=t_eval, rtol=1e-8, atol=1e-8)
    return sol.y.T
