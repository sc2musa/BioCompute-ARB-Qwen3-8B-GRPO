# Simulating stiff biochemical reaction networks

You are working in `/app`. Implement a general **mass-action reaction network simulator**
in `/app/simulate.py`. Hidden tests run it on several unseen networks (enzyme kinetics,
gene regulation, signalling cascades, epidemics written as reactions, random networks with
up to ~40 species). Several of them are **stiff**: their rate constants span many orders of
magnitude.

## Interface

```python
def simulate(network: dict, t_eval: list[float]) -> numpy.ndarray:
    """Return concentrations with shape (len(t_eval), n_species)."""
```

`t_eval` is increasing and starts at `0.0`; row `i` is the state at time `t_eval[i]`
(row 0 is the initial state). Column `j` corresponds to `network["species"][j]`.

## Network format (JSON-compatible dict)

```json
{
  "name": "michaelis_menten",
  "species": [{"name": "E", "initial": 0.01}, {"name": "S", "initial": 1.0}, ...],
  "reactions": [
    {"reactants": {"E": 1, "S": 1}, "products": {"C": 1}, "rate": 10000.0},
    {"reactants": {"P": 2}, "products": {"D": 1}, "rate": 50.0},
    {"reactants": {}, "products": {"M": 1}, "rate": 0.01},
    {"reactants": {"M": 1}, "products": {}, "rate": 0.5}
  ],
  "t_end": 500.0
}
```

Mass-action kinetics: a reaction with rate constant `k` and reactant stoichiometries
`{X: a, Y: b}` proceeds at rate `v = k * [X]**a * [Y]**b` (no combinatorial factors). It
changes each species by `(product coefficient - reactant coefficient) * v`. Empty
`reactants` means a zero-order source (`v = k`); empty `products` means degradation. A
species may appear on both sides (e.g. catalysts, gene templates).

## Accuracy and speed requirements

For every hidden network, every time point and every species:

```
|y - y_ref| <= 1e-4 * (|y_ref| + 1e-3 * max_t |y_ref(species)|)
```

where `y_ref` is a high-accuracy reference solution. Each network must finish within
**60 seconds** on one CPU core (most good solutions need well under one second).

## Environment and resources

- Python 3 with `numpy` and `scipy` installed; no internet access.
- `/app/examples/` contains two public networks together with their reference trajectories
  (`t_eval`, `reference`) so you can check your accuracy. Hidden networks are different.

## Success condition

The task is solved when all hidden networks meet the accuracy bound within the time limit.
The score is the fraction of hidden networks that pass.
