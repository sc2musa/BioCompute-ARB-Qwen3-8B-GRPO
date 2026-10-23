import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Extract network parameters
    species = network["species"]
    reactions = network["reactions"]
    t_end = network.get("t_end", t_eval[-1])
    
    # Create a map from species names to indices
    species_map = {s["name"] : i for i, s in enumerate(species)}
    
    # Initialize concentrations
    concentrations = np.array([s["initial"] for s in species])
    
    # Define the derivative function
    def dydt(t, y):
        dy = np.zeros_like(y)
        for reaction in reactions:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate
            rate_value = rate
            for species_name, stoichiometry in reactants.items():
                rate_value *= y[species_map[species_name]] ** stoichiometry
            
            # Update the derivatives
            for species_name, stoichiometry in products.items():
                idx = species_map[species_name]
                dy[idx] += stoichiometry * rate_value
            
            for species_name, stoichiometry in reactants.items():
                idx = species_map[species_name]
                dy[idx] -= stoichiometry * rate_value
        return dy
    
    # Solve the ODE system with tighter tolerances
    sol = solve_ivp(dydt, [0, t_end], concentrations, t_eval=t_eval, method="BDF",
                    rtol=1e-8, atol=1e-8)
    
    # Return the solution
    return sol.y.T