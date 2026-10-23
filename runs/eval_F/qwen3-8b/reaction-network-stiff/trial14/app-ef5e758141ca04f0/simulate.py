import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    species = network["species"]
    n_species = len(species)
    
    # Precompute species name to index mapping
    species_indices = {s["name"] : i for i, s in enumerate(species)}
    
    # Extract initial concentrations
    y0 = np.array([s["initial"] for s in species])
    
    # Define the ODE system
    def dydt(t, y):
        dy = np.zeros(n_species)
        
        # Iterate over reactions
        for reaction in network["reactions"]:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate of the reaction
            rate_reaction = rate
            for species_name, stoich in reactants.items():
                idx = species_indices[species_name]
                rate_reaction *= y[idx] ** stoich
            
            # Update the derivative for each species
            for species_name, stoich in products.items():
                idx = species_indices[species_name]
                dy[idx] += stoich * rate_reaction
            
            for species_name, stoich in reactants.items():
                idx = species_indices[species_name]
                dy[idx] -= stoich * rate_reaction
        
        return dy
    
    # Solve the ODE with adjusted solver settings for stiffness
    sol = solve_ivp(dydt, [0, network["t_end"]], y0, t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-8)
    
    # Return the solution
    return sol.y.T