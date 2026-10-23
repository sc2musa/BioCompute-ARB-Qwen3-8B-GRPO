import numpy as np
from scipy.integrate import solve_ivp


def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    # Parse the network into arrays
    species = network["species"]
    n_species = len(species)
    reactions = network["reactions"]
    n_reactions = len(reactions)

    # Create reactant-order matrix A and net stoichiometry matrix S
    A = np.zeros((n_reactions, n_species))
    S = np.zeros((n_species, n_reactions))

    for r, reaction in enumerate(reactions):
        reactants = reaction["reactants"]
        products = reaction["products"]
        rate = reaction["rate"]

        # Fill reactant-order matrix A
        for s, stoich in reactants.items():
            s_idx = next(i for i, sp in enumerate(species) if sp["name"] == s)
            A[r, s_idx] = stoich

        # Fill net stoichiometry matrix S
        for s, stoich in products.items():
            s_idx = next(i for i, sp in enumerate(species) if sp["name"] == s)
            S[s_idx, r] += stoich
        for s, stoich in reactants.items():
            s_idx = next(i for i, sp in enumerate(species) if sp["name"] == s)
            S[s_idx, r] -= stoich

    # Initial concentrations
    y0 = np.array([sp["initial"] for sp in species])

    # Define the ODE system
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            # Compute reaction rate
            rate = reactions[r]["rate"]
            # Compute product of concentrations raised to reactant orders
            product = 1.0
            for s, stoich in reactions[r]["reactants"].items():
                s_idx = next(i for i, sp in enumerate(species) if sp["name"] == s)
                product *= y[s_idx] ** stoich
            v[r] = rate * product

        # Compute dy/dt = S @ v
        return S @ v

    # Define the Jacobian
    def jac(t, y):
        J = np.zeros((n_species, n_reactions))
        for r in range(n_reactions):
            rate = reactions[r]["rate"]
            # Compute derivative of reaction rate with respect to each species
            for s in range(n_species):
                stoich = A[r, s]
                if stoich > 0:
                    # Derivative of v_r with respect to y_s
                    J[s, r] = rate * stoich * y[s] ** (stoich - 1) * np.prod([y[i] ** A[r, i] for i in range(n_species) if i != s])
        return J

    # Set tolerances
    rtol = 1e-8
    atol = 1e-12 * np.max(y0)

    # Solve the ODE
    sol = solve_ivp(
        fun=dydt,
        t_span=(0.0, network["t_end"]),
        y0=y0,
        t_eval=t_eval,
        method='BDF',
        jac=jac,
        rtol=rtol,
        atol=atol
    )

    # Return the solution
    return sol.y.T