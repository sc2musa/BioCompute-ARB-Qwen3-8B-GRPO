import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m+1, n+1), dtype=int)
    
    # Initialize first row and column
    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] + gap_open + (i-1)*gap_extend
    for j in range(1, n+1):
        dp[0][j] = dp[0][j-1] + gap_open + (j-1)*gap_extend
    
    # Fill DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Match/mismatch score
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Choose maximum score from three possibilities
            dp[i][j] = max(
                dp[i-1][j-1] + score,  # Match
                dp[i-1][j] + gap_extend,  # Gap in seq_b
                dp[i][j-1] + gap_extend   # Gap in seq_a
            )
    
    # Find the optimal alignment
    if mode == "global":
        score = dp[m][n]
        # Backtrack to find the alignment
        aligned_a = ["" for _ in range(m)]
        aligned_b = ["" for _ in range(n)]
        i, j = m, n
        while i > 0 and j > 0:
            if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a[i-1] = seq_a[i-1]
                aligned_b[j-1] = seq_b[j-1]
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i-1][j] + gap_extend:
                aligned_a[i-1] = seq_a[i-1]
                aligned_b[j-1] = "-"
                i -= 1
            else:
                aligned_a[i-1] = "-"
                aligned_b[j-1] = seq_b[j-1]
                j -= 1
        # Handle remaining characters
        while i > 0:
            aligned_a[i-1] = seq_a[i-1]
            aligned_b[i-1] = "-"
            i -= 1
        while j > 0:
            aligned_a[j-1] = "-"
            aligned_b[j-1] = seq_b[j-1]
            j -= 1
        return (score, "".join(aligned_a), "".join(aligned_b))
    else:  # mode == "local"
        # Find the maximum score in the DP table
        max_score = np.max(dp)
        if max_score <= 0:
            return (0, "", "")
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        # Backtrack to find the alignment
        aligned_a = ["" for _ in range(m)]
        aligned_b = ["" for _ in range(n)]
        while i > 0 and j > 0:
            if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a[i-1] = seq_a[i-1]
                aligned_b[j-1] = seq_b[j-1]
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i-1][j] + gap_extend:
                aligned_a[i-1] = seq_a[i-1]
                aligned_b[j-1] = "-"
                i -= 1
            else:
                aligned_a[i-1] = "-"
                aligned_b[j-1] = seq_b[j-1]
                j -= 1
        # Handle remaining characters
        while i > 0:
            aligned_a[i-1] = seq_a[i-1]
            aligned_b[i-1] = "-"
            i -= 1
        while j > 0:
            aligned_a[j-1] = "-"
            aligned_b[j-1] = seq_b[j-1]
            j -= 1
        return (max_score, "".join(aligned_a), "".join(aligned_b))