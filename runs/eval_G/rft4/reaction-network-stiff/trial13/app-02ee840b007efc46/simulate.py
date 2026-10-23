import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    reactions = network['reactions']
    n_species = len(species)
    species_dict = {s['name']: idx for idx, s in enumerate(species)}
    initial_conditions = [s['initial'] for s in species]

    def dydt(t, y):
        dydt = np.zeros(n_species)
        for reaction in reactions:
            reactants = reaction['reactants']
            products = reaction['products']
            rate = reaction['rate']
            product = 1.0
            for s, stoich in reactants.items():
                idx = species_dict[s]
                product *= y[idx] ** stoich
            rate *= product
            for s, stoich in reactants.items():
                idx = species_dict[s]
                dydt[idx] -= stoich * rate
            for s, stoich in products.items():
                idx = species_dict[s]
                dydt[idx] += stoich * rate
        return dydt

    sol = solve_ivp(fun=dydt, t_span=[0, network['t_end']], y0=initial_conditions, t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-12)
    return sol.y.T
