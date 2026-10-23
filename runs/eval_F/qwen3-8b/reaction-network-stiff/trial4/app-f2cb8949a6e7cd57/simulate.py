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
    
    # Create species name to index mapping
    species_index = {s["name"] : i for i, s in enumerate(species)}
    
    # Define the ODE system
    def dydt(t, y):
        dy = np.zeros(n_species)
        for reaction in reactions:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate
            rate_val = rate
            for species_name, stoich in reactants.items():
                rate_val *= y[species_index[species_name]] ** stoich
            
            # Update the derivative
            for species_name, stoich in products.items():
                idx = species_index[species_name]
                dy[idx] += stoich * rate_val
            for species_name, stoich in reactants.items():
                idx = species_index[species_name]
                dy[idx] -= stoich * rate_val
        return dy
    
    # Solve the ODE system with LSODA method and tighter tolerances
    sol = solve_ivp(dydt, [0, t_end], y0, t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-8)
    
    # Return the solution
    return sol.y.T