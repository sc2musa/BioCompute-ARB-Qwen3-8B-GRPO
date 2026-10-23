import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros(len_b + 1, dtype=int)
    
    # Initialize first row
    for j in range(1, len_b + 1):
        dp[j] = dp[j-1] + gap_open + (j-1) * gap_extend
    
    # Fill DP table
    for i in range(1, len_a + 1):
        # Save previous row
        prev_dp = dp.copy()
        
        # Update current row
        for j in range(1, len_b + 1):
            # Match/mismatch score
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Gap penalties
            gap_a = prev_dp[j] + gap_extend
            gap_b = dp[j-1] + gap_extend
            
            # Choose maximum score
            dp[j] = max(score, gap_a, gap_b)
    
    # Get maximum score
    max_score = dp[len_b] if mode == "global" else max(dp)
    
    # Traceback to get aligned sequences
    aligned_a = ["" for _ in range(len_a + len_b)]
    aligned_b = ["" for _ in range(len_a + len_b)]
    i, j = len_a, len_b
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[j] == prev_dp[j] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a[i-1] = seq_a[i-1]
            aligned_b[j-1] = seq_b[j-1]
            i -= 1
            j -= 1
        elif i > 0 and dp[j] == prev_dp[j] + gap_extend:
            aligned_a[i-1] = seq_a[i-1]
            aligned_b[j-1] = "-"
            i -= 1
        elif j > 0 and dp[j] == dp[j-1] + gap_extend:
            aligned_a[i-1] = "-"
            aligned_b[j-1] = seq_b[j-1]
            j -= 1
    
    # Join the aligned sequences
    aligned_a = "".join(aligned_a)
    aligned_b = "".join(aligned_b)
    
    return (max_score, aligned_a, aligned_b)