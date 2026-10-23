import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize first row and column
    for i in range(1, len_a + 1):
        dp[i][0] = dp[i-1][0] + gap_open + (i-1)*gap_extend
    for j in range(1, len_b + 1):
        dp[0][j] = dp[0][j-1] + gap_open + (j-1)*gap_extend
    
    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Match/mismatch
            match_score = dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            # Gap in seq_a
            gap_a = dp[i-1][j] + gap_open + (i-1)*gap_extend
            # Gap in seq_b
            gap_b = dp[i][j-1] + gap_open + (j-1)*gap_extend
            
            # Choose the maximum score
            dp[i][j] = max(match_score, gap_a, gap_b)
    
    # Find the optimal score
    if mode == "global":
        score = dp[len_a][len_b]
    else:
        # For local, find the maximum score in the table
        score = np.max(dp)
    
    # Trace back to get the alignment
    aligned_a, aligned_b = "", ""
    i, j = len_a, len_b
    while i > 0 and j > 0:
        if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            # Match/mismatch
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j] + gap_open + (i-1)*gap_extend:
            # Gap in seq_a
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
        else:
            # Gap in seq_b
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
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
    
    return (score, aligned_a, aligned_b)