import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    n_species = len(species)
    y0 = np.array([s['initial'] for s in species])
    species_index = {s['name']: idx for idx, s in enumerate(species)}

    def dydt(t, y):
        derivatives = np.zeros(n_species)
        for reaction in network['reactions']:
            rate = reaction['rate']
            # Multiply by reactants' concentrations^stoichiometry
            for name, stoich in reaction['reactants'].items():
                idx = species_index[name]
                rate *= y[idx] ** stoich
            # Update derivatives for reactants and products
            for name, stoich in reaction['reactants'].items():
                idx = species_index[name]
                derivatives[idx] -= stoich * rate
            for name, stoich in reaction['products'].items():
                idx = species_index[name]
                derivatives[idx] += stoich * rate
        return derivatives

    sol = solve_ivp(dydt, (0, network['t_end']), y0, t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-8)
    return sol.y.T
