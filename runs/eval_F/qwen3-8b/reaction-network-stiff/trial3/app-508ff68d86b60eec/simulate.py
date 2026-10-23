import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Extract network information
    species = network["species"]
    reactions = network["reactions"]
    n_species = len(species)
    
    # Create a dictionary to map species names to indices
    species_index = {species[i]["name"] : i for i in range(n_species)}
    
    # Initialize concentrations
    concentrations = np.array([s["initial"] for s in species])
    
    # Define the derivative function
    def dydt(t, y):
        dy = np.zeros(n_species)
        for reaction in reactions:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate
            rate_value = rate
            for species_name, stoichiometry in reactants.items():
                if species_name in species_index:
                    rate_value *= y[species_index[species_name]] ** stoichiometry
            
            # Update the derivatives
            for species_name, stoichiometry in products.items():
                if species_name in species_index:
                    index = species_index[species_name]
                    dy[index] += stoichiometry * rate_value
            for species_name, stoichiometry in reactants.items():
                if species_name in species_index:
                    index = species_index[species_name]
                    dy[index] -= stoichiometry * rate_value
        return dy
    
    # Solve the ODEs with BDF method for stiff systems
    sol = solve_ivp(dydt, [0, network["t_end"]], concentrations, t_eval=t_eval, method='BDF', atol=1e-8, rtol=1e-8)
    
    # Return the solution
    return sol.y.T