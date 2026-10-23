import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Extract species and reactions
    species = network["species"]
    reactions = network["reactions"]
    t_end = network["t_end"]
    
    # Initialize concentrations
    concentrations = np.array([s["initial"] for s in species])
    
    # Create a function to compute the reaction rates
    def reaction_rates(t, y):
        rates = np.zeros(len(reactions))
        for i, reaction in enumerate(reactions):
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Compute the rate
            rate_value = rate
            for species_name, stoichiometry in reactants.items():
                rate_value *= y[species.index(species_name)] ** stoichiometry
            
            rates[i] = rate_value
        return rates
    
    # Create a function to compute the derivative dy/dt
    def dydt(t, y):
        dy = np.zeros_like(y)
        for reaction in reactions:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Compute the rate
            rate_value = rate
            for species_name, stoichiometry in reactants.items():
                rate_value *= y[species.index(species_name)] ** stoichiometry
            
            # Update the derivative
            for species_name, stoichiometry in products.items():
                idx = species.index(species_name)
                dy[idx] += stoichiometry * rate_value
            for species_name, stoichiometry in reactants.items():
                idx = species.index(species_name)
                dy[idx] -= stoichiometry * rate_value
        return dy
    
    # Solve the ODE
    sol = solve_ivp(dydt, [0, t_end], concentrations, t_eval=t_eval, method='BDF')
    
    # Return the solution
    return sol.y.T