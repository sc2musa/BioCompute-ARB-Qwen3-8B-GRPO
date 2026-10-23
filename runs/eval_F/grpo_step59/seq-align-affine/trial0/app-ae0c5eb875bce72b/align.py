import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m+1, n+1), dtype=int)
    
    # Fill the DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Calculate the score for matching/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Calculate the score for gap extensions
            gap_score = gap_open + (i-1) * gap_extend if i > 1 else gap_open
            
            # Calculate the score for the current cell
            dp[i][j] = max(
                dp[i-1][j-1] + score,
                dp[i-1][j] + gap_score,
                dp[i][j-1] + gap_score
            )
    
    # Find the optimal score and position
    if mode == "global":
        score = dp[m][n]
        i, j = m, n
    else:
        # Find the maximum score in the table
        max_score = np.max(dp)
        # Find the position with the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        score = max_score
        
    # Backtrack to find the alignment
    aligned_a = ""
    aligned_b = ""
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
        elif dp[i][j-1] == dp[i][j]:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            j -= 1
        else:
            # This should not happen with proper implementation
            aligned_a = "-" + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
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