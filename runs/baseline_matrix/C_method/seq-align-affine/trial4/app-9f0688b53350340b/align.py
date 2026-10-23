import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize matrices
    n, m = len(seq_a), len(seq_b)
    M = np.full((n+1, m+1), -np.inf)
    X = np.full((n+1, m+1), -np.inf)
    Y = np.full((n+1, m+1), -np.inf)

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
        end_cell = np.unravel_index([M[n][m], X[n][m], Y[n][m]].argmax(), (n+1, m+1))
    elif mode == "local":
        best_score = M.max()
        end_cell = np.unravel_index(M.argmax(), (n+1, m+1))
        if best_score == 0:
            return (0, "", "")

    # Backtrack to find the alignment
    aligned_a, aligned_b = ",", ""
    i, j = end_cell
    while i > 0 and j > 0:
        if M[i][j] == max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1]):
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        elif X[i][j] == max(M[i-1][j] + gap_open, X[i-1][j] + gap_extend, Y[i-1][j] + gap_open):
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        elif Y[i][j] == max(M[i][j-1] + gap_open, Y[i][j-1] + gap_extend, X[i][j-1] + gap_open):
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
    
    # Handle remaining characters
    while i > 0:
        aligned_a = seq_a[i-1] + aligned_a
        aligned_b = "-" + aligned_b
        i -= 1
    while j > 0:
        aligned_a = "-" + aligned_a
        aligned_b = seq_b[j-1] + aligned_b
        j -= 1

    return (best_score, aligned_a, aligned_b)