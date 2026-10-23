import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    species = network["species"]
    reactions = network["reactions"]
    t_end = network["t_end"]
    n_species = len(species)
    
    # Initial concentrations
    y0 = np.array([s["initial"] for s in species])
    
    # Define the ODE system
    def dydt(t, y):
        # Compute the rate of change for each species
        dy = np.zeros(n_species)
        
        # Compute the rate of each reaction
        for reaction in reactions:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the reaction rate
            rate_reaction = rate
            for species_name, stoichiometry in reactants.items():
                idx = next(i for i, s in enumerate(species) if s["name"] == species_name)
                rate_reaction *= y[idx] ** stoichiometry
            
            # Update the concentration changes
            for species_name, stoichiometry in products.items():
                idx = next(i for i, s in enumerate(species) if s["name"] == species_name)
                dy[idx] += rate_reaction * stoichiometry
            
            for species_name, stoichiometry in reactants.items():
                idx = next(i for i, s in enumerate(species) if s["name"] == species_name)
                dy[idx] -= rate_reaction * stoichiometry
        
        return dy
    
    # Solve the ODE system with tighter tolerances
    sol = solve_ivp(dydt, [0, t_end], y0, t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-8)
    
    # Return the solution
    return sol.y.T