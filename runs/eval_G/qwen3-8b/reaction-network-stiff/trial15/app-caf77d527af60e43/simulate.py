import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    reactions = network['reactions']
    t_end = network['t_end']
    species_index = {s['name']: i for i, s in enumerate(species)}
    y0 = np.array([s['initial'] for s in species])

    def dydt(t, y):
        derivs = np.zeros_like(y)
        for reaction in reactions:
            rate = reaction['rate']
            reactants = reaction['reactants']
            products = reaction['products']
            # Calculate the product of concentrations for reactants
            for spec, stoich in reactants.items():
                if spec in species_index:
                    rate *= y[species_index[spec]] ** stoich
            # Update derivatives
            for spec, stoich in reactants.items():
                derivs[species_index[spec]] -= stoich * rate
            for spec, stoich in products.items():
                derivs[species_index[spec]] += stoich * rate
        return derivs

    sol = solve_ivp(dydt, [0, t_end], y0, t_eval=t_eval, method='LSODA', rtol=1e-6, atol=1e-6)
    return sol.y