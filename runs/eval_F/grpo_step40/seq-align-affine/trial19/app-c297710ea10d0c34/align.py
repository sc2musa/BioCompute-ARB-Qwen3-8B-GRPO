import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Handle empty sequences
    if not seq_a or not seq_b:
        return (0, seq_a, seq_b)
    
    # Initialize DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m+1, n+1))
    
    # Initialize first row and column
    for i in range(1, m+1):
        dp[i][0] = gap_open + (i-1)*gap_extend
    for j in range(1, n+1):
        dp[0][j] = gap_open + (j-1)*gap_extend
    
    # Fill DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Calculate score for match/mismatch
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Calculate gap penalties
            gap_a = dp[i-1][j] + gap_extend
            gap_b = dp[i][j-1] + gap_extend
            
            # Choose the maximum score
            dp[i][j] = max(score + dp[i-1][j-1], gap_a, gap_b)
    
    # Get the maximum score
    if mode == "global":
        max_score = dp[m][n]
    else:
        # For local alignment, find the maximum value in the table
        max_score = np.max(dp)
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        # Set the rest of the table to -infinity to avoid using them
        dp = np.where(dp != max_score, -np.inf, dp)
    
    # Reconstruct the alignment
    aligned_a = ""
    aligned_b = ""
    i, j = m, n
    if mode == "local":
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
    
    while i > 0 and j > 0:
        if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j] + gap_extend:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        else:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
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
    
    return (max_score, aligned_a, aligned_b)