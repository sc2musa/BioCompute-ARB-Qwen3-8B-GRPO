## Method guidance

Use the Gotoh dynamic programme with three score matrices of size `(n+1) x (m+1)`:

- `M[i][j]`: best score of an alignment of `a[:i]`, `b[:j]` ending with column `(a_i, b_j)`;
- `X[i][j]`: ... ending with column `(a_i, '-')` (gap in `b`);
- `Y[i][j]`: ... ending with column `('-', b_j)` (gap in `a`).

Recurrences (`s` = match/mismatch score):

```
M[i][j] = s(a_i, b_j) + max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1])
X[i][j] = max(M[i-1][j] + gap_open, X[i-1][j] + gap_extend, Y[i-1][j] + gap_open)
Y[i][j] = max(M[i][j-1] + gap_open, Y[i][j-1] + gap_extend, X[i][j-1] + gap_open)
```

- Global: `M[0][0] = 0`; `X[i][0] = gap_open + (i-1)*gap_extend`;
  `Y[0][j] = gap_open + (j-1)*gap_extend`; every other border cell is `-inf`. The answer is
  `max(M[n][m], X[n][m], Y[n][m])`. Take care with the borders: this is where most published
  implementations go wrong.
- Local: in the `M` recurrence also allow starting fresh (`max(0, ...)` over the previous
  cells); the answer is the maximum `M[i][j]` over all cells (or 0). Here every `X`/`Y` border
  cell is `-inf` (an optimal local alignment never starts or ends with a gap), and `M`
  border cells may be `0` or `-inf` (the fresh start covers both).
- Store a back-pointer per matrix and cell recording which matrix the maximum came from, then
  trace back from the best end cell, switching matrices as the pointers say. In local mode
  give the fresh start its own pointer value and **stop the traceback at that cell**;
  continuing past it adds columns that are not part of the optimal local alignment.
- Edge cases: if one sequence is empty, the global alignment is a single gap (score
  `gap_open + (k-1)*gap_extend`, or 0 for two empty sequences); local mode returns
  `(0, "", "")` when no cell scores above 0.
- Plain Python lists with local-variable lookups are fast enough for 1500 x 1500.

Validate before submitting: check `examples.json`, and compare against an exhaustive search
over all alignments for random sequences of length <= 5.
