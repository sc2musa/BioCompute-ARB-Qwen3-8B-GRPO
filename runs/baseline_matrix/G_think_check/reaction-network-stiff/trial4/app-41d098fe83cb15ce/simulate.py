import numpy as np
from scipy.integrate import solve_ivp
from collections import defaultdict

def simulate(network, t_eval):
    species = network['species']
    species_index = {s['name']: i for i, s in enumerate(species)}
    initial = np.array([s['initial'] for s in species])
    
    def dydt(t, y):
        dy = np.zeros_like(y)
        for reaction in network['reactions']:
            rate = reaction['rate']
            for species_name, stoich in reaction['reactants'].items():
                rate *= y[species_index[species_name]] ** stoich
            
            net_stoich = defaultdict(float)
            for species_name, stoich in reaction['reactants'].items():
                net_stoich[species_name] -= stoich
            for species_name, stoich in reaction['products'].items():
                net_stoich[species_name] += stoich
            
            for species_name, stoich in net_stoich.items():
                if species_name in species_index:
                    idx = species_index[species_name]
                    dy[idx] += stoich * rate
        return dy
    
    sol = solve_ivp(dydt, [0, network['t_end']], initial, t_eval=t_eval, method='Radau', rtol=1e-8, atol=1e-10)
    return sol.y.T
