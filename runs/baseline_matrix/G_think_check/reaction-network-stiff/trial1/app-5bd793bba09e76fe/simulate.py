import numpy as np
from scipy.integrate import solve_ivp
from collections import defaultdict

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    n_species = len(species)
    species_names = [s['name'] for s in species]
    species_index = {name: idx for idx, name in enumerate(species_names)}
    y0 = np.array([s['initial'] for s in species])

    def dydt(t, y):
        dydt = np.zeros(n_species)
        for reaction in network['reactions']:
            rate = reaction['rate']
            # Calculate the product of concentrations^stoichiometry for reactants
            for species_name, stoich in reaction['reactants'].items():
                if species_name in species_index:
                    idx = species_index[species_name]
                    rate *= y[idx] ** stoich
            # Compute net stoichiometry
            net_stoich = defaultdict(int)
            for species_name, stoich in reaction['reactants'].items():
                net_stoich[species_name] -= stoich
            for species_name, stoich in reaction['products'].items():
                net_stoich[species_name] += stoich
            # Apply to dydt
            for species_name, coeff in net_stoich.items():
                if species_name in species_index:
                    idx = species_index[species_name]
                    dydt[idx] += coeff * rate
        return dydt

    sol = solve_ivp(dydt, [0.0, network['t_end']], y0, t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-8)
    return sol.y.T