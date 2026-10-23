# Inferring transmission and an unknown intervention from outbreak data

You are working in `/app`. You are an epidemic modelling analyst. For each outbreak you
receive the **daily number of new cases** for the first 70 days. Somewhere during that
period a public-health intervention *may* have reduced transmission, but the day and
strength of the intervention are not recorded. Your job is to build an estimator in
`/app/fit.py` that recovers the transmission parameters, detects the intervention, and
forecasts the next weeks. Hidden tests run your estimator on unseen outbreaks.

## Data-generating assumptions (what is known)

- Closed population of size `population`; deterministic SEIR dynamics
  (Susceptible -> Exposed -> Infectious -> Removed) with
  latent period `latent_period` (= 1/sigma, days) and infectious period
  `infectious_period` (= 1/gamma, days). Both are **known** for each outbreak.
- At time 0: `initial_infectious` people are Infectious, nobody is Exposed or Removed,
  everyone else is Susceptible.
- Transmission rate `beta(t) = R0 * gamma` before the intervention and
  `R_post * gamma` from the intervention day onward; `R_post <= R0`. Some outbreaks have
  no intervention at all (`R_post == R0`).
- `cases[d-1]` is the number of new symptomatic cases (transitions Exposed -> Infectious)
  reported during day `d` (time interval `(d-1, d]`), for `d = 1..70`. Reporting is complete
  but noisy (over-dispersed count noise around the model expectation).

## Interface

```python
def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    return {
        "R0": float,               # basic reproduction number before the intervention
        "R_post": float,           # reproduction number after the intervention (= R0 if none)
        "change_day": float | None,# day the intervention started (None if you detect none)
        "forecast": list[float],   # expected new cases for days 71 .. 70 + horizon
    }
```

## Scoring (per hidden outbreak)

- `R0` and `R_post`: relative error <= 10%.
- `change_day`: absolute error <= 3 days (only scored for clear interventions where
  `R_post <= 0.75 * R0`).
- `forecast`: `sum|forecast - expected| / sum(expected) <= 0.30`, compared with the
  noise-free expected incidence of the true model.

An outbreak passes when all its scored criteria hold. The task is solved when at least
10 of the 12 hidden outbreaks pass; the continuous score is the mean fraction of criteria met.

## Resources

- Python 3 with `numpy` and `scipy`; no internet access. At most 120 seconds per outbreak on
  one CPU core.
- `/app/data/example*.json` contains three public outbreaks with their inputs (`inputs`)
  and the ground truth (`truth`), so you can validate your estimator. Hidden outbreaks
  differ in all parameters.
