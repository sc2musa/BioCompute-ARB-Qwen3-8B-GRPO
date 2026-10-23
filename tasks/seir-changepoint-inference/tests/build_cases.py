"""Author-side script: generate synthetic outbreaks with a hidden intervention.

Not used at grading time. Run from the repo root:
    python tasks/seir-changepoint-inference/tests/build_cases.py
Writes tests/cases.json (hidden data + ground truth) and public examples with truth into
environment/workspace/data/.

Model: SEIR with known population N, latent period 1/sigma, infectious period 1/gamma and
initial infectious I0 (E0 = R0 = 0). Transmission rate beta0 before day t_c, beta1 after.
Observed daily cases = new E->I transitions per day, with negative-binomial noise.
"""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

TASK_DIR = Path(__file__).resolve().parents[1]
SEED = 20261022
N_OBS_DAYS = 70
HORIZON = 28
NB_DISPERSION = 30.0  # variance = mu + mu^2 / k
N_HIDDEN = 12
N_NO_INTERVENTION = 3


def expected_incidence(population, i0, latent, infectious, r0, r_post, change_day, n_days):
    """Expected new cases on days 1..n_days (integral of sigma*E over each day)."""
    sigma, gamma = 1.0 / latent, 1.0 / infectious

    def rhs(t, y):
        s, e, i, _c = y
        beta = (r0 if t < change_day else r_post) * gamma
        infection = beta * s * i / population
        return [-infection, infection - sigma * e, sigma * e - gamma * i, sigma * e]

    y0 = [population - i0, 0.0, i0, 0.0]
    t_eval = np.arange(n_days + 1, dtype=float)
    # Integrate the two regimes separately so the discontinuity in beta is resolved exactly
    tc = float(np.clip(change_day, 0.0, n_days))
    parts = []
    for t0, t1 in ((0.0, tc), (tc, float(n_days))):
        if t1 <= t0:
            continue
        pts = t_eval[(t_eval >= t0) & (t_eval <= t1)]
        sol = solve_ivp(rhs, (t0, t1), y0, method="Radau", rtol=1e-10, atol=1e-8,
                        t_eval=pts, dense_output=True)
        parts.append((pts, sol.y[3]))
        y0 = sol.sol(t1)
    t_all = np.concatenate([p[0] for p in parts])
    c_all = np.concatenate([p[1] for p in parts])
    _, keep = np.unique(t_all, return_index=True)
    cumulative = c_all[keep]
    return np.diff(cumulative)


def sample_case(rng, case_id, intervention=True):
    for _ in range(1000):
        population = int(10 ** rng.uniform(5, 7))
        i0 = float(rng.integers(5, 50))
        latent, infectious = rng.uniform(2.0, 5.0), rng.uniform(3.0, 8.0)
        r0 = rng.uniform(1.8, 3.2)
        total_days = N_OBS_DAYS + HORIZON
        free = expected_incidence(population, i0, latent, infectious, r0, r0, 1e9, total_days)
        # Intervention while the outbreak is clearly visible but before its peak
        visible = np.where((free >= 30) & (free <= 0.7 * free.max()))[0] + 1
        visible = visible[(visible >= 15) & (visible <= N_OBS_DAYS - 15)]
        if len(visible) < 5:
            continue
        if intervention:
            change_day = int(rng.choice(visible))
            r_post = r0 * rng.uniform(0.25, 0.9)
        else:
            change_day, r_post = None, r0
        mu = expected_incidence(population, i0, latent, infectious, r0, r_post,
                                change_day if intervention else 1e9, total_days)
        p = NB_DISPERSION / (NB_DISPERSION + mu)
        cases = rng.negative_binomial(NB_DISPERSION, p)
        return {
            "id": case_id,
            "inputs": {
                "cases": cases[:N_OBS_DAYS].astype(int).tolist(),
                "population": population,
                "initial_infectious": i0,
                "latent_period": round(latent, 4),
                "infectious_period": round(infectious, 4),
                "horizon": HORIZON,
            },
            "truth": {
                "R0": r0, "R_post": r_post, "change_day": change_day,
                "expected_forecast": mu[N_OBS_DAYS:].tolist(),
                "future_observed": cases[N_OBS_DAYS:].astype(int).tolist(),
            },
        }
    raise RuntimeError("could not sample a visible outbreak")


def main():
    rng = np.random.default_rng(SEED)
    hidden = [sample_case(rng, f"outbreak{k:02d}", intervention=k >= N_NO_INTERVENTION)
              for k in range(N_HIDDEN)]
    rng.shuffle(hidden)
    (TASK_DIR / "tests" / "cases.json").write_text(json.dumps(hidden, indent=1))
    pub_rng = np.random.default_rng(SEED + 1)
    data_dir = TASK_DIR / "environment" / "workspace" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for k, has_iv in enumerate([True, True, False]):
        case = sample_case(pub_rng, f"example{k}", intervention=has_iv)
        (data_dir / f"example{k}.json").write_text(json.dumps(case, indent=1))
    for c in hidden:
        t = c["truth"]
        print(c["id"], f"R0={t['R0']:.2f} Rpost={t['R_post']:.2f} tc={t['change_day']}",
              "max cases", max(c["inputs"]["cases"]))


if __name__ == "__main__":
    main()
