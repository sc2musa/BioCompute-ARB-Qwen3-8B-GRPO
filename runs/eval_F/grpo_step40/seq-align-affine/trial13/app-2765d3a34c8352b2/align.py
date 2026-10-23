import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    # Precompute gap scores
    gap_scores = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                gap_scores[i][j] = 0
            elif i == 0:
                gap_scores[i][j] = gap_open + (j - 1) * gap_extend
            elif j == 0:
                gap_scores[i][j] = gap_open + (i - 1) * gap_extend
            else:
                gap_scores[i][j] = gap_open + (i - 1) * gap_extend
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Calculate the score for matching characters
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Calculate the maximum score
            dp[i][j] = max(
                dp[i-1][j] + gap_scores[i][j],  # Gap in seq_a
                dp[i][j-1] + gap_scores[i][j],  # Gap in seq_b
                dp[i-1][j-1] + score     # Match
            )
    
    # Find the optimal score
    if mode == "global":
        score = dp[m][n]
        # Backtrack to find the alignment
        aligned_a, aligned_b = "", ""
        i, j = m, n
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif dp[i-1][j] + gap_scores[i][j] == dp[i][j]:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
            elif dp[i][j-1] + gap_scores[i][j] == dp[i][j]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                j -= 1
            else:
                # This case should not happen for global mode
                pass
        
        # Handle remaining characters
        while i > 0:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
        while j > 0:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            j -= 1
    else:
        # For local mode, find the maximum score in the DP table
        score = np.max(dp)
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        # Backtrack to find the alignment
        aligned_a, aligned_b = "", ""
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif dp[i-1][j] + gap_scores[i][j] == dp[i][j]:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
            elif dp[i][j-1] + gap_scores[i][j] == dp[i][j]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                j -= 1
            else:
                # This case should not happen for local mode
                pass
        
        # Handle remaining characters
        while i > 0:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
        while j > 0:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            j -= 1
    
    return (score, aligned_a, aligned_b)