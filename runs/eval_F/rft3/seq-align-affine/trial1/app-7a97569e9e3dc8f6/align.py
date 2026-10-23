import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the DP table
    len_a = len(seq_a)
    len_b = len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1))
    
    # Initialize the first row and column
    for i in range(1, len_a + 1):
        dp[i][0] = dp[i-1][0] + gap_open + (i-1) * gap_extend
    for j in range(1, len_b + 1):
        dp[0][j] = dp[0][j-1] + gap_open + (j-1) * gap_extend
    
    # Fill the DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate the score for matching/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Calculate the score for gap in seq_a
            gap_a = gap_open + (i-1) * gap_extend
            # Calculate the score for gap in seq_b
            gap_b = gap_open + (j-1) * gap_extend
            
            # Choose the maximum score
            dp[i][j] = max(
                dp[i-1][j-1] + score,
                dp[i-1][j] + gap_a,
                dp[i][j-1] + gap_b
            )
    
    # Find the maximum score and its position
    max_score = dp[len_a][len_b]
    i, j = len_a, len_b
    
    # Reconstruct the alignment
    aligned_a = ["" for _ in range(len_a + len_b)]
    aligned_b = ["" for _ in range(len_a + len_b)]
    
    while i > 0 and j > 0:
        # Check if we have a match/mismatch
        if seq_a[i-1] == seq_b[j-1]:
            aligned_a[i-1] = seq_a[i-1]
            aligned_b[j-1] = seq_b[j-1]
            i -= 1
            j -= 1
        # Check if we have a gap in seq_a
        elif dp[i-1][j] + gap_open + (i-1) * gap_extend == dp[i][j]:
            aligned_a[i-1] = "-"
            aligned_b[j-1] = seq_b[j-1]
            i -= 1
        # Check if we have a gap in seq_b
        elif dp[i][j-1] + gap_open + (j-1) * gap_extend == dp[i][j]:
            aligned_a[i-1] = seq_a[i-1]
            aligned_b[j-1] = "-"
            j -= 1
        # If none of the above, we have a gap in both sequences
        else:
            aligned_a[i-1] = "-"
            aligned_b[j-1] = "-"
            i -= 1
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
    
    # Return the results
    return (max_score, ''.join(aligned_a), ''.join(aligned_b))