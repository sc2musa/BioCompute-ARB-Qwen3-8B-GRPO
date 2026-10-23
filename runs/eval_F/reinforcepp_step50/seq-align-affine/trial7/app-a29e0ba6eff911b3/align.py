import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize DP table
    len_a = len(seq_a)
    len_b = len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Fill DP table
    for i in range(len_a + 1):
        for j in range(len_b + 1):
            if i == 0 and j == 0:
                dp[i][j] = 0
            elif i == 0:
                # Gap in seq_a
                dp[i][j] = dp[i][j-1] + gap_extend
            elif j == 0:
                # Gap in seq_b
                dp[i][j] = dp[i-1][j] + gap_extend
            else:
                # Calculate score for matching/mismatch
                if seq_a[i-1] == seq_b[j-1]:
                    score = match
                else:
                    score = mismatch
                
                # Calculate score for gap in seq_a
                gap_a = dp[i-1][j] + gap_open + (i-1)*gap_extend
                
                # Calculate score for gap in seq_b
                gap_b = dp[i][j-1] + gap_open + (j-1)*gap_extend
                
                # Choose maximum score
                dp[i][j] = max(score + dp[i-1][j-1], gap_a, gap_b)
    
    # Find the optimal score
    if mode == "global":
        score = dp[len_a][len_b]
    else:
        # Find the maximum score in the DP table
        score = np.max(dp)
        
    # Backtrack to find the alignment
    aligned_a = ""
    aligned_b = ""
    i, j = len_a, len_b
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq_a[i-1] == seq_b[j-1]:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + gap_open + (i-1)*gap_extend:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + gap_open + (j-1)*gap_extend:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
        else:
            # This should not happen for valid inputs
            break
    
    return (score, aligned_a, aligned_b)