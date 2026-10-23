import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    reactions = network['reactions']
    t_end = network['t_end']

    # Extract initial concentrations
    y0 = [s['initial'] for s in species]

    # Map species names to indices
    species_names = [s['name'] for s in species]
    name_to_index = {name: idx for idx, name in enumerate(species_names)}

    # Define the ODE function
    def dydt(t, y):
        dy = np.zeros_like(y)
        for reaction in reactions:
            reactants = reaction['reactants']
            products = reaction['products']
            rate = reaction['rate']
            # Calculate the rate v
            v = rate
            for species_name, stoich in reactants.items():
                idx = name_to_index[species_name]
                v *= y[idx] ** stoich
            # Calculate the contribution of this reaction to each species
            for species_name, stoich in products.items():
                idx = name_to_index[species_name]
                dy[idx] += stoich * v
            for species_name, stoich in reactants.items():
                idx = name_to_index[species_name]
                dy[idx] -= stoich * v
        return dy

    # Solve the ODE
    sol = solve_ivp(dydt, [0, t_end], y0, t_eval=t_eval, method='BDF', rtol=1e-6, atol=1e-6)
    return sol.y
