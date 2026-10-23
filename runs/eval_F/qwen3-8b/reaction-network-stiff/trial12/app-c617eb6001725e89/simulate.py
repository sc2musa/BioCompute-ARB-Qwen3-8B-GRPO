import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Extract species and reactions
    species = network["species"]
    n_species = len(species)
    
    # Create a dictionary to map species names to indices
    species_index = {species[i]["name"] : i for i in range(n_species)}
    
    # Initial concentrations
    y0 = np.array([s["initial"] for s in species])
    
    # Define the ODE system
    def reaction_rates(t, y):
        rates = np.zeros(len(network["reactions"]))
        for i, reaction in enumerate(network["reactions"]):
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate
            rate_value = rate
            for species_name, stoich in reactants.items():
                rate_value *= y[species_index[species_name]] ** stoich
            
            # Update the rates
            rates[i] = rate_value
        return rates
    
    # Define the ODE function
    def dydt(t, y):
        dy = np.zeros(n_species)
        for reaction in network["reactions"]:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate
            rate_value = rate
            for species_name, stoich in reactants.items():
                rate_value *= y[species_index[species_name]] ** stoich
            
            # Update the derivatives
            for species_name, stoich in products.items():
                idx = species_index[species_name]
                dy[idx] += stoich * rate_value
            for species_name, stoich in reactants.items():
                idx = species_index[species_name]
                dy[idx] -= stoich * rate_value
        return dy
    
    # Solve the ODE with BDF method for stiff systems and tighter tolerances
    sol = solve_ivp(dydt, [t_eval[0], network["t_end"]], y0, t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-8)
    
    # Return the solution
    return sol.y.T