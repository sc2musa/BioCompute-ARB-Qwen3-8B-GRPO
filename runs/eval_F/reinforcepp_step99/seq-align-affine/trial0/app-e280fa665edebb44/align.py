import numpy as np

def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Fill the first row and column with gap penalties
    for i in range(1, len_a + 1):
        dp[i][0] = dp[i-1][0] + gap_open
    for j in range(1, len_b + 1):
        dp[0][j] = dp[0][j-1] + gap_open
    
    # Fill the DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate the score for a match/mismatch
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            # Calculate the score for a gap in seq_a
            gap_a = dp[i-1][j] + gap_extend
            # Calculate the score for a gap in seq_b
            gap_b = dp[i][j-1] + gap_extend
            # Choose the maximum score
            dp[i][j] = max(score, gap_a, gap_b)
    
    # Extract the score
    score = dp[len_a][len_b]
    
    # Backtrack to find the aligned sequences
    aligned_a = ""
    aligned_b = ""
    i, j = len_a, len_b
    while i > 0 and j > 0:
        if seq_a[i-1] == seq_b[j-1]:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        elif dp[i-1][j] == dp[i][j]:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
        else:
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