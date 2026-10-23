import numpy as np
from scipy.integrate import solve_ivp


def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    """Return concentrations with shape (len(t_eval), n_species)."""
    
    # Extract network parameters
    species = network["species"]
    reactions = network["reactions"]
    n_species = len(species)
    
    # Initial concentrations
    y0 = np.array([s["initial"] for s in species])
    
    # Define the ODE system
    def dydt(t, y):
        dy = np.zeros(n_species)
        
        # Calculate the rate of each reaction
        for reaction in reactions:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the reaction rate
            rate_value = rate
            for species_name, stoichiometry in reactants.items():
                idx = next(i for i, s in enumerate(species) if s["name"] == species_name)
                rate_value *= y[idx] ** stoichiometry
            
            # Update the derivative for each species
            for species_name, stoichiometry in products.items():
                idx = next(i for i, s in enumerate(species) if s["name"] == species_name)
                dy[idx] += rate_value * stoichiometry
            
            for species_name, stoichiometry in reactants.items():
                idx = next(i for i, s in enumerate(species) if s["name"] == species_name)
                dy[idx] -= rate_value * stoichiometry
        
        return dy
    
    # Solve the ODE system
    sol = solve_ivp(dydt, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, method='BDF', dense_output=True, rtol=1e-6, atol=1e-6)
    
    # Extract the solution at the specified time points
    y = sol.sol(t_eval)
    
    return y.T