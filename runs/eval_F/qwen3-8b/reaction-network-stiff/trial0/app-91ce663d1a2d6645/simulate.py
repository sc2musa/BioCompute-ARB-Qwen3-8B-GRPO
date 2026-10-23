import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    # Extract species and initial concentrations
    species = network["species"]
    n_species = len(species)
    initial_concentrations = np.array([s["initial"] for s in species])

    # Create a dictionary to map species names to indices
    species_to_index = {species[i]["name"]: i for i in range(n_species)}

    # Extract reactions
    reactions = network["reactions"]

    # Define the ODE system
    def reaction_rates(t, y):
        rates = np.zeros(len(reactions))
        for i, reaction in enumerate(reactions):
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            # Calculate the rate for this reaction
            rate_value = rate
            for species_name, stoichiometry in reactants.items():
                rate_value *= y[species_to_index[species_name]] ** stoichiometry
            rates[i] = rate_value
        return rates

    # Define the derivative function
    def dydt(t, y):
        dy = np.zeros(n_species)
        rates = reaction_rates(t, y)
        for i, reaction in enumerate(reactions):
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = rates[i]
            # Update the derivative for each species
            for species_name, stoichiometry in products.items():
                idx = species_to_index[species_name]
                dy[idx] += stoichiometry * rate
            for species_name, stoichiometry in reactants.items():
                idx = species_to_index[species_name]
                dy[idx] -= stoichiometry * rate
        return dy

    # Solve the ODE with BDF method and adjusted tolerances
    sol = solve_ivp(dydt, (t_eval[0], t_eval[-1]), initial_concentrations, t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-8)

    # Return the solution
    return sol.y.T