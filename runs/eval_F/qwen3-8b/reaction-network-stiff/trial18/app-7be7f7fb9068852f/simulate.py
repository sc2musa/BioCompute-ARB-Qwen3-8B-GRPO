import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Extract network parameters
    species = network["species"]
    reactions = network["reactions"]
    t_end = network.get("t_end", 100.0)
    
    # Create a mapping from species names to indices
    species_names = [s["name"] for s in species]
    species_index = {name: i for i, name in enumerate(species_names)}
    
    # Initialize concentrations
    n_species = len(species)
    concentrations = np.array([s["initial"] for s in species])
    
    # Define the ODE function
    def dydt(t, y):
        dy = np.zeros(n_species)
        for reaction in reactions:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate
            rate_value = rate
            for species_name, stoich in reactants.items():
                rate_value *= y[species_index[species_name]] ** stoich
            
            # Update the concentrations
            for species_name, stoich in products.items():
                idx = species_index[species_name]
                dy[idx] += stoich * rate_value
            
            for species_name, stoich in reactants.items():
                idx = species_index[species_name]
                dy[idx] -= stoich * rate_value
        return dy
    
    # Solve the ODE with tighter tolerances
    sol = solve_ivp(dydt, [0, t_end], concentrations, t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-8)
    
    # Return the solution
    return sol.y.T