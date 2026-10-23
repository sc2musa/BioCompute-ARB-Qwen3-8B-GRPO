import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    n, m = len(seq_a), len(seq_b)
    
    # Initialize score matrices
    M = np.full((n+1, m+1), -np.inf, dtype=int)
    X = np.full((n+1, m+1), -np.inf, dtype=int)
    Y = np.full((n+1, m+1), -np.inf, dtype=int)
    
    # Initialize borders for global mode
    if mode == "global":
        M[0][0] = 0
        for i in range(1, n+1):
            X[i][0] = gap_open + (i-1)*gap_extend
        for j in range(1, m+1):
            Y[0][j] = gap_open + (j-1)*gap_extend
    
    # Initialize borders for local mode
    elif mode == "local":
        for i in range(n+1):
            for j in range(m+1):
                M[i][j] = max(0, M[i][j])
    
    # Fill the matrices
    for i in range(1, n+1):
        for j in range(1, m+1):
            # Calculate the score for M[i][j]
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            M[i][j] = score + max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1])
            
            # Calculate the score for X[i][j]
            X[i][j] = max(
                M[i-1][j] + gap_open,
                X[i-1][j] + gap_extend,
                Y[i-1][j] + gap_open
            )
            
            # Calculate the score for Y[i][j]
            Y[i][j] = max(
                M[i][j-1] + gap_open,
                Y[i][j-1] + gap_extend,
                X[i][j-1] + gap_open
            )
    
    # Determine the best score based on mode
    if mode == "global":
        best_score = max(M[n][m], X[n][m], Y[n][m])
    elif mode == "local":
        best_score = max(M)
    else:
        raise ValueError("Invalid mode.")
    
    # Backtrack to find the alignment
    # ... (backtracking code to be added)
    
    # Return the score, aligned_a, and aligned_b
    return (best_score, "", "")
