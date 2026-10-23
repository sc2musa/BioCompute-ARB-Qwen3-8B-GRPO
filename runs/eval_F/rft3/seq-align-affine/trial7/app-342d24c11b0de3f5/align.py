import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m+1, n+1), dtype=int)
    
    # Initialize first row and column
    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] + gap_open
    for j in range(1, n+1):
        dp[0][j] = dp[0][j-1] + gap_open
    
    # Fill DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Calculate score for match/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Calculate score for gap in seq_a
            score_a = dp[i-1][j] + gap_extend
            
            # Calculate score for gap in seq_b
            score_b = dp[i][j-1] + gap_extend
            
            # Choose the best score
            dp[i][j] = max(score, score_a, score_b)
    
    # Traceback to find the optimal alignment
    i, j = m, n
    aligned_a, aligned_b = "", ""
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq_a[i-1] == seq_b[j-1]:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + gap_extend:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + gap_extend:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
        else:
            # This should not happen for valid inputs
            aligned_a = "-" + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
            j -= 1
    
    return dp[m][n], aligned_a, aligned_b