import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Extract species information
    species = network["species"]
    n_species = len(species)
    
    # Create a mapping from species names to indices
    species_indices = {s["name"] : i for i, s in enumerate(species)}
    
    # Initialize concentrations
    concentrations = np.array([s["initial"] for s in species])
    
    # Extract reaction information
    reactions = network["reactions"]
    
    # Define the derivative function
    def dydt(t, y):
        dy = np.zeros(n_species)
        for reaction in reactions:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate
            rate_value = rate
            for species_name, stoich in reactants.items():
                rate_value *= y[species_indices[species_name]] ** stoich
            
            # Update the derivative for each species
            for species_name, stoich in products.items():
                idx = species_indices[species_name]
                dy[idx] += stoich * rate_value
            
            for species_name, stoich in reactants.items():
                idx = species_indices[species_name]
                dy[idx] -= stoich * rate_value
        return dy
    
    # Solve the ODE system with BDF method for stiff systems
    sol = solve_ivp(
        dydt, 
        [t_eval[0], t_eval[-1]], 
        concentrations, 
        t_eval=t_eval, 
        method='BDF', 
        rtol=1e-8, 
        atol=1e-8
    )
    
    # Return the solution
    return sol.y.T