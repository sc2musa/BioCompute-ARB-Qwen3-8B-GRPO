import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    n, m = len(seq_a), len(seq_b)
    
    # Initialize matrices
    M = np.full((n+1, m+1), -np.inf, dtype=int)
    X = np.full((n+1, m+1), -np.inf, dtype=int)
    Y = np.full((n+1, m+1), -np.inf, dtype=int)
    
    # Initialize borders based on mode
    if mode == "global":
        M[0][0] = 0
        for i in range(1, n+1):
            X[i][0] = gap_open + (i-1)*gap_extend
        for j in range(1, m+1):
            Y[0][j] = gap_open + (j-1)*gap_extend
    elif mode == "local":
        # Local mode: initialize all to 0 and handle later
        pass
    
    # Fill matrices
    for i in range(1, n+1):
        for j in range(1, m+1):
            # Calculate score for M[i][j]
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            M[i][j] = score + max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1])
            
            # Calculate X[i][j]
            X[i][j] = max(
                M[i-1][j] + gap_open,  # Gap in b
                X[i-1][j] + gap_extend,  # Gap in b
                Y[i-1][j] + gap_open  # Gap in a followed by gap in b
            )
            
            # Calculate Y[i][j]
            Y[i][j] = max(
                M[i][j-1] + gap_open,  # Gap in a
                Y[i][j-1] + gap_extend,  # Gap in a
                X[i][j-1] + gap_open  # Gap in b followed by gap in a
            )
    
    # Determine the best score based on mode
    if mode == "global":
        best_score = max(M[n][m], X[n][m], Y[n][m])
    elif mode == "local":
        best_score = max(M)
        if best_score < 0:
            best_score = 0
    
    # Traceback to find the alignment
    # This part is not implemented yet
    
    # Return the score and empty strings for now
    return (best_score, "", "")