import numpy as np
from scipy.integrate import solve_ivp


def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    """Return concentrations with shape (len(t_eval), n_species)."""
    
    # Extract species information
    species = network["species"]
    n_species = len(species)
    
    # Create a dictionary to map species names to indices
    species_indices = {species[i]["name"] : i for i in range(n_species)}
    
    # Initial concentrations
    y0 = np.array([s["initial"] for s in species])
    
    # Define the ODE system
    def reaction_rates(t, y):
        rates = np.zeros(len(network["reactions"]))
        
        # Calculate reaction rates
        for i, reaction in enumerate(network["reactions"]):
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate
            rate_value = rate
            for species_name, stoich in reactants.items():
                if species_name in species_indices:
                    rate_value *= y[species_indices[species_name]] ** stoich
            
            rates[i] = rate_value
        
        # Calculate the change in concentrations
        dydt = np.zeros(n_species)
        for i, reaction in enumerate(network["reactions"]):
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = rates[i]
            
            # Calculate the change for each species
            for species_name, stoich in products.items():
                if species_name in species_indices:
                    idx = species_indices[species_name]
                    dydt[idx] += stoich * rate
            
            for species_name, stoich in reactants.items():
                if species_name in species_indices:
                    idx = species_indices[species_name]
                    dydt[idx] -= stoich * rate
        
        return dydt
    
    # Solve the ODE system
    sol = solve_ivp(
        reaction_rates,
        [t_eval[0], t_eval[-1]],
        y0,
        t_eval=t_eval,
        method='BDF',
        rtol=1e-8,
        atol=1e-8
    )
    
    # Return the solution
    return sol.y.T
