import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Extract species and reactions
    species = network["species"]
    reactions = network["reactions"]
    t_end = network["t_end"]
    
    # Create a dictionary to map species names to indices
    species_indices = {species[i]["name"] : i for i in range(len(species))}
    
    # Initialize concentrations
    concentrations = np.array([s["initial"] for s in species])
    
    # Define the ODE system
    def dydt(t, y):
        dy = np.zeros_like(y)
        for reaction in reactions:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate
            rate_value = rate
            for species_name, stoichiometry in reactants.items():
                rate_value *= y[species_indices[species_name]] ** stoichiometry
            
            # Update the derivative for each species
            for species_name, stoichiometry in products.items():
                idx = species_indices[species_name]
                dy[idx] += stoichiometry * rate_value
            
            for species_name, stoichiometry in reactants.items():
                idx = species_indices[species_name]
                dy[idx] -= stoichiometry * rate_value
        return dy
    
    # Solve the ODE with tighter tolerances and a suitable method
    sol = solve_ivp(dydt, [0, t_end], concentrations, t_eval=t_eval, method='Radau', rtol=1e-6, atol=1e-6)
    
    # Return the solution
    return sol.y.T