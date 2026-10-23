import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m+1, n+1), dtype=int)
    
    # Initialize first row and column
    for i in range(1, m+1):
        dp[i][0] = gap_open + (i-1)*gap_extend
    for j in range(1, n+1):
        dp[0][j] = gap_open + (j-1)*gap_extend
    
    # Fill DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Calculate scores for all possible operations
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            insert_score = dp[i-1][j] + gap_extend
            delete_score = dp[i][j-1] + gap_extend
            
            # Choose the maximum score
            dp[i][j] = max(match_score + dp[i-1][j-1], insert_score, delete_score)
    
    # Trace back to find the optimal alignment
    i, j = m, n
    aligned_a, aligned_b = [], []
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + gap_extend:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        else:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
    
    # Reverse the alignments
    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]
    
    # Return the result
    return dp[m][n], "".join(aligned_a), "".join(aligned_b)