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
    
    # Create a mapping from species name to index
    species_index = {s["name"] : i for i, s in enumerate(species)}
    
    # Define the derivative function
    def dydt(t, y):
        dy = np.zeros(n_species)
        for reaction in reactions:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the reaction rate
            rate_value = rate
            for species_name, stoichiometry in reactants.items():
                rate_value *= y[species_index[species_name]] ** stoichiometry
            
            # Update the derivative for each species
            for species_name, stoichiometry in products.items():
                dy[species_index[species_name]] += stoichiometry * rate_value
            for species_name, stoichiometry in reactants.items():
                dy[species_index[species_name]] -= stoichiometry * rate_value
        return dy
    
    # Solve the ODE system
    sol = solve_ivp(dydt, [0, t_end], concentrations, t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-8)
    
    # Return the solution
    return sol.y.T
