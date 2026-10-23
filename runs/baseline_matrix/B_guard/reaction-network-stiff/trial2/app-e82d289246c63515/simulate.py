import numpy as np
from scipy.integrate import solve_ivp


def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    """Return concentrations with shape (len(t_eval), n_species)."""
    
    # Extract network parameters
    species = network["species"]
    reactions = network["reactions"]
    t_end = network["t_end"]
    
    # Initialize concentrations
    n_species = len(species)
    concentrations = np.array([s["initial"] for s in species])
    
    # Define the ODE system
    def reaction_rates(t, y):
        rates = np.zeros(len(reactions))
        for i, reaction in enumerate(reactions):
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate
            rate_value = rate
            for species_name, stoichiometry in reactants.items():
                rate_value *= y[species.index(species_name)] ** stoichiometry
            
            rates[i] = rate_value
        return rates
    
    # Define the derivative function
    def dydt(t, y):
        dy = np.zeros(n_species)
        
        # Calculate reaction rates
        rates = reaction_rates(t, y)
        
        # Update concentrations
        for i, reaction in enumerate(reactions):
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = rates[i]
            
            # Calculate the change in each species
            for species_name, stoichiometry in reactants.items():
                idx = species.index(species_name)
                dy[idx] -= stoichiometry * rate
            
            for species_name, stoichiometry in products.items():
                idx = species.index(species_name)
                dy[idx] += stoichiometry * rate
        
        return dy
    
    # Solve the ODEs
    sol = solve_ivp(
        fun=dydt,
        t_span=[0, t_end],
        y0=concentrations,
        t_eval=t_eval,
        method='BDF',
        rtol=1e-8,
        atol=1e-8
    )
    
    # Return the solution
    return sol.y.T