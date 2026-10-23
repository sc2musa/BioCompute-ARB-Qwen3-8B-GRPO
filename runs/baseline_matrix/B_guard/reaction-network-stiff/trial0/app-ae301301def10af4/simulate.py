import numpy as np
from scipy.integrate import solve_ivp


def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    """Return concentrations with shape (len(t_eval), n_species)."""
    
    # Extract species information
    species = network["species"]
    n_species = len(species)
    
    # Extract reaction information
    reactions = network["reactions"]
    
    # Create a function to compute the derivative
    def dydt(t, y):
        """Compute the derivative of concentrations."""
        dy = np.zeros(n_species)
        
        # Calculate the rate of each reaction
        for reaction in reactions:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate of the reaction
            rate_reaction = rate
            for species_name, stoichiometry in reactants.items():
                rate_reaction *= y[species.index(species_name)] ** stoichiometry
            
            # Update the derivative for each species
            for species_name, stoichiometry in products.items():
                idx = species.index(species_name)
                dy[idx] += stoichiometry * rate_reaction
            
            for species_name, stoichiometry in reactants.items():
                idx = species.index(species_name)
                dy[idx] -= stoichiometry * rate_reaction
        
        return dy
    
    # Initial concentrations
    y0 = np.array([species["initial"] for species in species])
    
    # Solve the ODE system
    sol = solve_ivp(dydt, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, method='BDF')
    
    # Return the solution
    return sol.y.T