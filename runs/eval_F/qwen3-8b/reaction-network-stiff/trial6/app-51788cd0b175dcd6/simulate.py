import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Extract network parameters
    species = network["species"]
    reactions = network["reactions"]
    t_end = network.get("t_end", 100.0)
    
    # Initialize concentrations
    n_species = len(species)
    y0 = np.array([s["initial"] for s in species])
    
    # Create a dictionary to map species names to indices
    species_index = {species[i]["name"] : i for i in range(n_species)}
    
    # Define the ODE system
    def reaction_rates(t, y):
        rates = np.zeros(len(reactions))
        for i, reaction in enumerate(reactions):
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate
            rate_value = rate
            for species_name, stoich in reactants.items():
                rate_value *= y[species_index[species_name]] ** stoich
            
            # Calculate the change in concentrations
            delta_y = np.zeros(n_species)
            for species_name, stoich in products.items():
                idx = species_index[species_name]
                delta_y[idx] += stoich * rate_value
            for species_name, stoich in reactants.items():
                idx = species_index[species_name]
                delta_y[idx] -= stoich * rate_value
            
            rates[i] = rate_value
            
        return delta_y
    
    # Solve the ODE
    sol = solve_ivp(reaction_rates, [0, t_end], y0, t_eval=t_eval, method='LSODA', rtol=1e-10, atol=1e-10)
    
    # Return the solution
    return sol.y.T