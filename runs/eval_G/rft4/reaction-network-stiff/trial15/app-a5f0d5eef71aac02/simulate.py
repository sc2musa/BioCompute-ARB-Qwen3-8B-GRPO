import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    n_species = len(species)
    species_names = [s['name'] for s in species]
    species_index = {name: idx for idx, name in enumerate(species_names)}

    # Create A and B matrices
    reactions = network['reactions']
    n_reactions = len(reactions)
    A = np.zeros((n_reactions, n_species))
    B = np.zeros((n_reactions, n_species))

    for r_idx, reaction in enumerate(reactions):
        reactants = reaction['reactants']
        products = reaction['products']
        rate = reaction['rate']

        # Fill A matrix (reactants)
        for s_name, stoich in reactants.items():
            s_idx = species_index[s_name]
            A[r_idx, s_idx] = stoich

        # Fill B matrix (products)
        for s_name, stoich in products.items():
            s_idx = species_index[s_name]
            B[r_idx, s_idx] = stoich

    # Initial concentrations
    y0 = np.array([s['initial'] for s in species])

    # Define the ODE function
    def dydt(t, y):
        dy = np.zeros(n_species)
        for r_idx in range(n_reactions):
            # Compute the product of y^A[r_idx][s] for each species
            product = np.prod(y ** A[r_idx, :])
            # Compute the rate for this reaction
            rate = reactions[r_idx]['rate']
            v_r = rate * product
            # Multiply by (B - A) for each species
            dy += (B[r_idx, :] - A[r_idx, :]) * v_r
        return dy

    # Solve the ODE using BDF method with tight tolerances
    sol = solve_ivp(
        fun=dydt,
        t_span=[0.0, network['t_end']],
        y0=y0,
        t_eval=t_eval,
        method='BDF',
        rtol=1e-8,
        atol=1e-12
    )

    # Return the solution as a numpy array with shape (len(t_eval), n_species)
    return sol.y.T
