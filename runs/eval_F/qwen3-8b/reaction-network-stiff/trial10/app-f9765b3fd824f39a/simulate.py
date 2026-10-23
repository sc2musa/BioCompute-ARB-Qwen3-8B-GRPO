import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Extract species information
    species = network["species"]
    n_species = len(species)
    
    # Create a dictionary to map species names to indices
    species_indices = {species[i]["name"] : i for i in range(n_species)}
    
    # Initialize concentrations
    concentrations = np.array([s["initial"] for s in species])
    
    # Define the ODE system
    def reaction_rates(t, y):
        rates = np.zeros(n_species)
        
        # Calculate reaction rates
        for reaction in network["reactions"]:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate of this reaction
            rate_value = rate
            for species_name, stoichiometry in reactants.items():
                rate_value *= y[species_indices[species_name]] ** stoichiometry
            
            # Update the rates for species involved in this reaction
            for species_name, stoichiometry in products.items():
                index = species_indices[species_name]
                rates[index] += rate_value * stoichiometry
            
            for species_name, stoichiometry in reactants.items():
                index = species_indices[species_name]
                rates[index] -= rate_value * stoichiometry
        
        return rates
    
    # Solve the ODE system using Radau method for stiff systems
    sol = solve_ivp(
        reaction_rates,
        [t_eval[0], t_eval[-1]],
        concentrations,
        t_eval=t_eval,
        method='Radau',
        rtol=1e-6,
        atol=1e-6
    )
    
    # Return the solution at the specified time points
    return sol.y.T
