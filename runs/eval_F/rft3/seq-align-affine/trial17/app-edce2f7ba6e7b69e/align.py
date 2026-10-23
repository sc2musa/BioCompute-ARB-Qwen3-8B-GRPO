import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    len_a, len_b = len(seq_a), len(seq_b)
    
    # Initialize the DP table
    dp = np.zeros(len_b + 1)
    
    # Initialize the first column
    for j in range(1, len_b + 1):
        dp[j] = dp[j-1] + gap_open + (j-1)*gap_extend
    
    # Fill the DP table
    for i in range(1, len_a + 1):
        # Calculate the score for the current row
        current_row = np.zeros(len_b + 1)
        
        # Calculate the score for the first column
        current_row[0] = dp[0] + gap_open + (i-1)*gap_extend
        
        # Calculate the score for the other columns
        for j in range(1, len_b + 1):
            # Calculate the score for matching characters
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Calculate the score for gaps
            gap_a = dp[j] + gap_extend
            gap_b = current_row[j-1] + gap_extend
            
            # Choose the best score
            current_row[j] = max(match_score + dp[j-1], gap_a, gap_b)
        
        # Update the DP table
        dp = current_row
    
    # Extract the best score
    best_score = dp[len_b]
    
    # Backtrack to find the alignment
    aligned_a = ""
    aligned_b = ""
    i, j = len_a, len_b
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[j] == match_score + dp[j-1]:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        elif i > 0 and dp[j] == dp[j] + gap_extend:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        elif j > 0 and dp[j] == dp[j-1] + gap_extend:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
    
    return (best_score, aligned_a, aligned_b)