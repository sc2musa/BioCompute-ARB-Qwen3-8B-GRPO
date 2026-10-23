# Pairwise sequence alignment with affine gap penalties

You are working in `/app`. Implement **optimal pairwise alignment of two DNA sequences with
affine gap penalties** in `/app/align.py`.

## Interface

```python
def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
```

- `mode="global"`: Needleman–Wunsch style; the alignment covers both full sequences.
- `mode="local"`: Smith–Waterman style; the alignment covers one substring of each
  sequence and has the maximum score over all substring pairs. If no positive-scoring local
  alignment exists, return `(0, "", "")`.

## Scoring model

- An aligned column `(x, y)` with two letters scores `match` if `x == y`, else `mismatch`.
- A **gap** is a maximal run of consecutive columns that all have `-` in the *same*
  sequence. A gap of length `k` scores `gap_open + (k - 1) * gap_extend`
  (both parameters are negative integers; `gap_open` is the score of the first gap column).
- A gap in `seq_a` directly followed by a gap in `seq_b` counts as two separate gaps.
- A column with `-` in both sequences is not allowed.
- The alignment score is the sum over columns/gaps and must be the **maximum** possible.

`aligned_a` / `aligned_b` are the two rows of one optimal alignment (equal length, `-` for
gaps). Removing the `-` characters must give back the full inputs (global mode) or
substrings of them (local mode), and re-scoring the returned rows must give `score`.
When several optimal alignments exist, any one is accepted.

## Requirements

- Pure Python 3 standard library and `numpy` only. Ready-made alignment libraries
  (Biopython, parasail, edlib, scikit-bio, ...) are not allowed.
- Sequences may be empty. Lengths up to 1500 x 1500 must finish within 60 seconds
  per call on a single CPU core.
- `/app/examples.json` contains a few public examples with their optimal scores; hidden
  tests use different sequences and scoring parameters.

## Success condition

Your solution passes when every hidden test case returns the optimal score together with a
valid alignment that re-scores to that value, and the 1500 x 1500 cases meet the time limit.
