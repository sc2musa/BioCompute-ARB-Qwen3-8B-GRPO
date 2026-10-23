import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Extract network parameters
    species = network["species"]
    reactions = network["reactions"]
    t_end = network["t_end"]
    
    # Initialize concentrations
    n_species = len(species)
    concentrations = np.array([s["initial"] for s in species])
    
    # Create a dictionary to map species names to indices
    species_indices = {species[i]["name"] : i for i in range(n_species)}
    
    # Define the derivative function
    def dydt(t, y):
        dy = np.zeros(n_species)
        for reaction in reactions:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the reaction rate
            rate_value = rate
            for species_name, stoich in reactants.items():
                idx = species_indices[species_name]
                rate_value *= y[idx] ** stoich
            
            # Update the derivative for each species
            for species_name, stoich in products.items():
                idx = species_indices[species_name]
                dy[idx] += stoich * rate_value
            
            for species_name, stoich in reactants.items():
                idx = species_indices[species_name]
                dy[idx] -= stoich * rate_value
        return dy
    
    # Solve the ODE system with tighter tolerances
    sol = solve_ivp(dydt, [0, t_end], concentrations, t_eval=t_eval, method='BDF', 
                    rtol=1e-8, atol=1e-8)
    
    # Return the solution
    return sol.y.T