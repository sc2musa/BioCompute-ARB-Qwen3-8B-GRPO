import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    """Return concentrations with shape (len(t_eval), n_species)."""
    
    # Extract species and reactions
    species = network["species"]
    n_species = len(species)
    
    # Initialize concentrations
    concentrations = np.array([s["initial"] for s in species])
    
    # Define the ODE system
    def reaction_rates(t, y):
        rates = np.zeros(len(network["reactions"]))
        for i, reaction in enumerate(network["reactions"]):
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate of the reaction
            rate_value = rate
            for species_name, stoichiometry in reactants.items():
                rate_value *= y[species.index(species_name)] ** stoichiometry
            
            # Update the rates
            rates[i] = rate_value
        return rates
    
    # Define the derivative function
    def dydt(t, y):
        dy = np.zeros(n_species)
        for reaction in network["reactions"]:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate of the reaction
            rate_value = rate
            for species_name, stoichiometry in reactants.items():
                rate_value *= y[species.index(species_name)] ** stoichiometry
            
            # Update the derivatives
            for species_name, stoichiometry in products.items():
                idx = species.index(species_name)
                dy[idx] += stoichiometry * rate_value
            
            for species_name, stoichiometry in reactants.items():
                idx = species.index(species_name)
                dy[idx] -= stoichiometry * rate_value
        return dy
    
    # Solve the ODE system
    sol = solve_ivp(dydt, [0, network["t_end"]], concentrations, t_eval=t_eval, method='LSODA')
    
    # Return the solution
    return sol.y.T