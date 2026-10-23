import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    species = network["species"]
    n_species = len(species)
    
    # Extract initial concentrations
    y0 = np.array([s["initial"] for s in species])
    
    # Define the ODE system
    def dydt(t, y):
        rates = []
        for reaction in network["reactions"]:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate of the reaction
            rate_val = rate
            for species_name, stoich in reactants.items():
                rate_val *= y[species.index(species_name)] ** stoich
            
            # Calculate the change in concentrations
            dy = np.zeros(n_species)
            for species_name, stoich in products.items():
                idx = species.index(species_name)
                dy[idx] += stoich * rate_val
            for species_name, stoich in reactants.items():
                idx = species.index(species_name)
                dy[idx] -= stoich * rate_val
            
            rates.append(rate_val)
        
        # Return the derivative of concentrations
        return dy
    
    # Solve the ODE system
    sol = solve_ivp(dydt, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval, method='LSODA')
    
    # Return the solution
    return sol.y.T
