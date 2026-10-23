import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m+1, n+1), dtype=int)
    
    # Fill DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Calculate scores for all possible operations
            match_score = dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            gap_a = dp[i-1, j] + gap_open + (i-1)*gap_extend
            gap_b = dp[i, j-1] + gap_open + (j-1)*gap_extend
            
            # Choose the maximum score
            dp[i, j] = max(match_score, gap_a, gap_b)
    
    # Find the maximum score in the table
    max_score = dp.max()
    
    # Find the position of the maximum score
    i, j = np.unravel_index(dp.argmax(), dp.shape)
    
    # Traceback to find the optimal alignment
    aligned_a, aligned_b = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i, j] == dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i-1, j] + gap_open + (i-1)*gap_extend:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        else:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
    
    # Reverse the alignment
    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]
    
    # Return the score and aligned sequences
    return (max_score, "".join(aligned_a), "".join(aligned_b))