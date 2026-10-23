import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m+1, n+1))
    
    # Fill the first row and column
    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] + gap_open
    for j in range(1, n+1):
        dp[0][j] = dp[0][j-1] + gap_open
    
    # Fill the DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Calculate the score for matching/mismatch
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Calculate the score for gap in seq_a
            score_a = dp[i-1][j] + gap_extend
            
            # Calculate the score for gap in seq_b
            score_b = dp[i][j-1] + gap_extend
            
            # Choose the maximum score
            dp[i][j] = max(score, score_a, score_b)
    
    # Get the maximum score
    max_score = dp[m][n] if mode == "global" else np.max(dp)
    
    # Reconstruct the alignment
    aligned_a, aligned_b = "", ""
    i, j = m, n
    while i > 0 and j > 0:
        if dp[i][j] == dp[i-1][j] + gap_extend:
            aligned_a += "-"
            aligned_b += seq_b[j-1]
            j -= 1
        elif dp[i][j] == dp[i][j-1] + gap_extend:
            aligned_a += seq_a[i-1]
            aligned_b += "-"
            i -= 1
        else:
            aligned_a += seq_a[i-1]
            aligned_b += seq_b[j-1]
            i -= 1
            j -= 1
    
    # Handle remaining characters
    while i > 0:
        aligned_a += seq_a[i-1]
        aligned_b += "-"
        i -= 1
    while j > 0:
        aligned_a += "-"
        aligned_b += seq_b[j-1]
        j -= 1
    
    # Reverse the alignment
    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]
    
    return (max_score, aligned_a, aligned_b)