import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Extract network parameters
    species = network["species"]
    reactions = network["reactions"]
    n_species = len(species)
    
    # Create a mapping from species name to index
    species_map = {s["name"] : i for i, s in enumerate(species)}
    
    # Initial concentrations
    y0 = np.array([s["initial"] for s in species])
    
    # Define the ODE system
    def dydt(t, y):
        # Compute the rate of change for each species
        dy = np.zeros(n_species)
        
        # Calculate the contribution from each reaction
        for reaction in reactions:
            # Get reactant and product stoichiometries
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the reaction rate
            rate_expr = 1.0
            for species_name, stoich in reactants.items():
                idx = species_map[species_name]
                rate_expr *= y[idx] ** stoich
            
            # Calculate the change in concentrations
            for species_name, stoich in products.items():
                idx = species_map[species_name]
                dy[idx] += rate * stoich
            
            for species_name, stoich in reactants.items():
                idx = species_map[species_name]
                dy[idx] -= rate * stoich
        
        return dy
    
    # Solve the ODE system with appropriate settings for stiff problems
    sol = solve_ivp(dydt, [0, network["t_end"]], y0, t_eval=t_eval, method='BDF',
                    rtol=1e-8, atol=1e-8, dense_output=True, max_step=0.000001)
    
    # Return the solution
    return sol.y.T
