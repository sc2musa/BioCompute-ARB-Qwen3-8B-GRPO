import numpy as np
from scipy.integrate import solve_ivp


def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    """Return concentrations with shape (len(t_eval), n_species)."""
    
    # Extract species information
    species = network["species"]
    n_species = len(species)
    
    # Create a mapping from species names to indices
    species_index = {species[i]["name"] : i for i in range(n_species)}
    
    # Extract reaction information
    reactions = network["reactions"]
    
    # Initialize concentrations
    concentrations = np.array([s["initial"] for s in species])
    
    # Define the ODE system
    def dydt(t, y):
        dy = np.zeros(n_species)
        for reaction in reactions:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate of this reaction
            rate_value = rate
            for species_name, stoichiometry in reactants.items():
                rate_value *= y[species_index[species_name]] ** stoichiometry
            
            # Update the derivative for each species
            for species_name, stoichiometry in products.items():
                idx = species_index[species_name]
                dy[idx] += stoichiometry * rate_value
            
            for species_name, stoichiometry in reactants.items():
                idx = species_index[species_name]
                dy[idx] -= stoichiometry * rate_value
        return dy
    
    # Solve the ODE system with BDF method for stiff systems
    sol = solve_ivp(
        dydt,
        [t_eval[0], t_eval[-1]],
        concentrations,
        t_eval=t_eval,
        method='BDF',
        rtol=1e-8,
        atol=1e-8,
        dense_output=True
    )
    
    # Return the solution
    return sol.y.T
