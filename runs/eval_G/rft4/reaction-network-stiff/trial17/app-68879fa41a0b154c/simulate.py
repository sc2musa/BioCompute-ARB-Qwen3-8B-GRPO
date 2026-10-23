import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network["species"]
    reactions = network["reactions"]
    n_species = len(species)
    n_reactions = len(reactions)

    species_names = [s["name"] for s in species]
    species_index = {name: idx for idx, name in enumerate(species_names)}

    A = np.zeros((n_reactions, n_species))
    S = np.zeros((n_species, n_reactions))

    for r in range(n_reactions):
        reactants = reactions[r]["reactants"]
        products = reactions[r]["products"]
        rate = reactions[r]["rate"]

        # Fill A matrix (reactant stoichiometry)
        for s in reactants:
            A[r, species_index[s]] = reactants[s]

        # Fill S matrix (net stoichiometry)
        for s in products:
            S[species_index[s], r] += products[s]
        for s in reactants:
            S[species_index[s], r] -= reactants[s]

    # Initial conditions
    y0 = np.array([s["initial"] for s in species])

    # Define ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            rate = reactions[r]["rate"]
            product = np.prod(y ** A[r, :])
            v[r] = rate * product
        return S @ v

    # Solve ODE
    sol = solve_ivp(dydt, [0, network["t_end"]], y0, t_eval=t_eval, method='BDF', 
                    rtol=1e-8, atol=1e-12)

    return sol.y.T