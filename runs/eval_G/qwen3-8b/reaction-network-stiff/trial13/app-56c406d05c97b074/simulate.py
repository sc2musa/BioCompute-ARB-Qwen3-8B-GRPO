import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    initial = np.array([s['initial'] for s in species])
    species_to_index = {s['name']: i for i, s in enumerate(species)}
    
    def dydt(t, y):
        dy = np.zeros_like(y)
        for reaction in network['reactions']:
            rate = reaction['rate']
            # Compute rate
            for reactant, stoich in reaction['reactants'].items():
                rate *= y[species_to_index[reactant]] ** stoich
            # Compute net stoichiometry
            net_stoich = {}
            for reactant, stoich in reaction['reactants'].items():
                net_stoich[reactant] = -stoich
            for product, stoich in reaction['products'].items():
                net_stoich[product] = net_stoich.get(product, 0) + stoich
            # Update dy
            for species_name, stoich in net_stoich.items():
                dy[species_to_index[species_name]] += stoich * rate
        return dy
    
    t_span = (0.0, network['t_end'])
    sol = solve_ivp(dydt, t_span, initial, t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-9)
    return sol.y.T
