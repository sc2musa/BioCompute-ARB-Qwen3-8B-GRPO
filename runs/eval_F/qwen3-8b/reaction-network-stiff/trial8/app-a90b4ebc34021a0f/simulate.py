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
            
            # Calculate the rate
            reactant_concentrations = np.array([y[species_index[name]] for name in reactants])
            rate_value = rate * np.prod(reactant_concentrations ** np.array(list(reactants.values())))
            
            # Update the derivative
            for species_name, stoichiometry in products.items():
                idx = species_index[species_name]
                dy[idx] += stoichiometry * rate_value
            for species_name, stoichiometry in reactants.items():
                idx = species_index[species_name]
                dy[idx] -= stoichiometry * rate_value
        return dy
    
    # Solve the ODE system with tighter tolerances
    sol = solve_ivp(dydt, [0, t_end], concentrations, t_eval=t_eval, method='BDF', dense_output=True, rtol=1e-8, atol=1e-8)
    
    # Extract the solution
    return sol.y.T