import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Parse network
    species = network["species"]
    n_species = len(species)
    reactions = network["reactions"]
    n_reactions = len(reactions)

    # Create reactant order matrix A (reactions x species)
    A = np.zeros((n_reactions, n_species))
    for r, reaction in enumerate(reactions):
        for s_name, stoich in reaction["reactants"].items():
            s = next(i for i, s in enumerate(species) if s["name"] == s_name)
            A[r, s] = stoich

    # Create net stoichiometry matrix S (species x reactions)
    S = np.zeros((n_species, n_reactions))
    for r, reaction in enumerate(reactions):
        for s_name, stoich in reaction["products"].items():
            s = next(i for i, s in enumerate(species) if s["name"] == s_name)
            S[s, r] += stoich
        for s_name, stoich in reaction["reactants"].items():
            s = next(i for i, s in enumerate(species) if s["name"] == s_name)
            S[s, r] -= stoich

    # Initial concentrations
    y0 = np.array([s["initial"] for s in species])

    # Define ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            # Calculate reaction rate
            rate = reactions[r]["rate"]
            for s in range(n_species):
                if A[r, s] > 0:
                    rate *= y[s] ** A[r, s]
            v[r] = rate
        # Calculate dy/dt
        return S @ v

    # Set tolerances
    rtol = 1e-8
    atol = 1e-12 * np.max(y0) if np.max(y0) > 0 else 1e-12

    # Solve ODE
    sol = solve_ivp(dydt, [0, t_eval[-1]], y0, t_eval=t_eval, method="BDF",
                    rtol=rtol, atol=atol, dense_output=True)

    # Return solution
    return sol.y.T