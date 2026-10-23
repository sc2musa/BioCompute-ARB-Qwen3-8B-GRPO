import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1))
    
    # Initialize the first row and column for global mode
    if mode == "global":
        for i in range(1, len_a + 1):
            dp[i][0] = dp[i-1][0] + gap_open
        for j in range(1, len_b + 1):
            dp[0][j] = dp[0][j-1] + gap_open
    
    # Fill the DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate the score for matching characters
            score_match = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Calculate the score for inserting a gap in seq_a
            score_gap_a = dp[i-1][j] + gap_extend
            
            # Calculate the score for inserting a gap in seq_b
            score_gap_b = dp[i][j-1] + gap_extend
            
            # Choose the maximum score
            dp[i][j] = max(score_match, score_gap_a, score_gap_b)
    
    # Extract the score
    score = dp[len_a][len_b]
    
    # Backtrack to find the aligned sequences
    aligned_a = ["-" for _ in range(len_a + len_b)]
    aligned_b = ["-" for _ in range(len_a + len_b)]
    i, j = len_a, len_b
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a[i-1] = seq_a[i-1]
            aligned_b[j-1] = seq_b[j-1]
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + gap_extend:
            aligned_a[i-1] = seq_a[i-1]
            aligned_b[j-1] = "-"
            i -= 1
        else:
            aligned_a[i-1] = "-"
            aligned_b[j-1] = seq_b[j-1]
            j -= 1
    
    # Remove the extra "-" characters
    aligned_a = [c for c in aligned_a if c != "-"]
    aligned_b = [c for c in aligned_b if c != "-"]
    
    return (score, "".join(aligned_a), "".join(aligned_b))