import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1))
    
    # Initialize the first row and column
    for i in range(1, len_a + 1):
        dp[i, 0] = dp[i-1, 0] + gap_open + (i-1) * gap_extend
    for j in range(1, len_b + 1):
        dp[0, j] = dp[0, j-1] + gap_open + (j-1) * gap_extend
    
    # Fill the DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate the score for matching/mismatch
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Calculate the score for gaps
            gap_score = gap_open + (i-1) * gap_extend
            
            # Calculate the maximum score
            dp[i, j] = max(
                dp[i-1, j] + gap_score,  # Gap in seq_a
                dp[i, j-1] + gap_score,  # Gap in seq_b
                dp[i-1, j-1] + score      # Match/mismatch
            )
    
    # Extract the optimal alignment
    i, j = len_a, len_b
    aligned_a, aligned_b = ",", ""
    
    while i > 0 and j > 0:
        if dp[i, j] == dp[i-1, j] + gap_open + (i-1) * gap_extend:
            aligned_a = "-" + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        elif dp[i, j] == dp[i, j-1] + gap_open + (j-1) * gap_extend:
            aligned_a = "-" + aligned_a
            aligned_b = "-" + aligned_b
            j -= 1
        else:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
    
    # Handle remaining characters
    while i > 0:
        aligned_a = "-" + aligned_a
        aligned_b = "-" + aligned_b
        i -= 1
    while j > 0:
        aligned_a = "-" + aligned_a
        aligned_b = "-" + aligned_b
        j -= 1
    
    # Return the result
    return (dp[len_a, len_b], aligned_a, aligned_b)