import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    species_index = {s['name']: i for i, s in enumerate(species)}
    reactions = network['reactions']
    y0 = [s['initial'] for s in species]

    def dydt(t, y):
        dydt = np.zeros_like(y)
        for reaction in reactions:
            reactants = reaction['reactants']
            products = reaction['products']
            rate = reaction['rate']
            product = 1.0
            for species_name, stoich in reactants.items():
                species_idx = species_index[species_name]
                product *= y[species_idx] ** stoich
            rate *= product
            species_in_reaction = set(reactants.keys()) | set(products.keys())
            for species_name in species_in_reaction:
                product_stoich = products.get(species_name, 0)
                reactant_stoich = reactants.get(species_name, 0)
                species_idx = species_index[species_name]
                dydt[species_idx] += (product_stoich - reactant_stoich) * rate
        return dydt

    t_span = (0.0, t_eval[-1])
    sol = solve_ivp(dydt, t_span, y0, t_eval=t_eval, method='LSODA', rtol=1e-9, atol=1e-9)
    return sol.y.T
