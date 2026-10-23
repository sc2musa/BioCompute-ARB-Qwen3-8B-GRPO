import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize matrices
    n, m = len(seq_a), len(seq_b)
    M = np.full((n+1, m+1), -np.inf, dtype=int)
    X = np.full((n+1, m+1), -np.inf, dtype=int)
    Y = np.full((n+1, m+1), -np.inf, dtype=int)

    # Initialize borders
    if mode == "global":
        M[0][0] = 0
        for i in range(1, n+1):
            X[i][0] = gap_open + (i-1)*gap_extend
        for j in range(1, m+1):
            Y[0][j] = gap_open + (j-1)*gap_extend
    elif mode == "local":
        for i in range(n+1):
            for j in range(m+1):
                M[i][j] = max(0, M[i][j])

    # Fill matrices
    for i in range(1, n+1):
        for j in range(1, m+1):
            # Calculate score for M[i][j]
            s = match if seq_a[i-1] == seq_b[j-1] else mismatch
            M[i][j] = s + max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1])

            # Calculate score for X[i][j]
            X[i][j] = max(
                M[i-1][j] + gap_open,
                X[i-1][j] + gap_extend,
                Y[i-1][j] + gap_open
            )

            # Calculate score for Y[i][j]
            Y[i][j] = max(
                M[i][j-1] + gap_open,
                Y[i][j-1] + gap_extend,
                X[i][j-1] + gap_open
            )

    # Determine the best score and end cell
    if mode == "global":
        best_score = max(M[n][m], X[n][m], Y[n][m])
        best_cell = (n, m) if M[n][m] == best_score else (n, m) if X[n][m] == best_score else (n, m)
    elif mode == "local":
        best_score = max(M)
        best_cell = np.unravel_index(np.argmax(M), M.shape)

    # Backtrack to find the alignment
    aligned_a = []
    aligned_b = []
    i, j = best_cell
    while i > 0 and j > 0:
        if M[i][j] == max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1]):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif X[i][j] == max(M[i-1][j] + gap_open, X[i-1][j] + gap_extend, Y[i-1][j] + gap_open):
            aligned_a.append(seq_a[i-1])
            aligned_b.append('-')
            i -= 1
        elif Y[i][j] == max(M[i][j-1] + gap_open, Y[i][j-1] + gap_extend, X[i][j-1] + gap_open):
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            j -= 1
    
    # Reverse the alignment to get the correct order
    aligned_a = ''.join(reversed(aligned_a))
    aligned_b = ''.join(reversed(aligned_b))

    # Return the result
    return (best_score, aligned_a, aligned_b)
