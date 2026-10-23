import numpy
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> numpy.ndarray:
    species = network['species']
    n_species = len(species)
    y0 = numpy.array([s['initial'] for s in species])
    reactions = network['reactions']
    n_reactions = len(reactions)
    rates = [r['rate'] for r in reactions]

    # Build A and S matrices
    A = [[0.0 for _ in range(n_species)] for _ in range(n_reactions)]
    S = [[0.0 for _ in range(n_reactions)] for _ in range(n_species)]
    for r in range(n_reactions):
        reactants = reactions[r]['reactants']
        products = reactions[r]['products']
        for s in range(n_species):
            name = species[s]['name']
            A[r][s] = reactants.get(name, 0)
            product_count = products.get(name, 0)
            S[s][r] = product_count - A[r][s]

    # Define ODE function
    def ode(t, y):
        v = numpy.zeros(n_reactions)
        for r in range(n_reactions):
            product = 1.0
            for s in range(n_species):
                product *= y[s] ** A[r][s]
            v[r] = rates[r] * product
        dydt = numpy.zeros(n_species)
        for s in range(n_species):
            for r in range(n_reactions):
                dydt[s] += S[s][r] * v[r]
        return dydt

    # Define Jacobian function
    def jac(t, y, dydt):
        v = numpy.zeros(n_reactions)
        for r in range(n_reactions):
            product = 1.0
            for s in range(n_species):
                product *= y[s] ** A[r][s]
            v[r] = rates[r] * product
        J = numpy.zeros((n_species, n_species))
        for s in range(n_species):
            for s_prime in range(n_species):
                total = 0.0
                for r in range(n_reactions):
                    if y[s_prime] == 0:
                        continue
                    term = S[s][r] * A[r][s_prime] * v[r] / y[s_prime]
                    total += term
                J[s][s_prime] = total
        return J

    # Set tolerances
    max_initial = max(y0)
    atol = 1e-12 * max_initial
    rtol = 1e-8

    # Solve ODE
    sol = solve_ivp(ode, [0.0, network['t_end']], y0, t_eval=t_eval, method='BDF', jac=jac, rtol=rtol, atol=atol)

    # Return concentrations
    return sol.y.T