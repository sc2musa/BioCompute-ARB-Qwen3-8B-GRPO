import numpy as np


def align(seq_a, seq_b, mode="global",
          match=2, mismatch=-3,
          gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Handle empty sequences
    if not seq_a or not seq_b:
        return (0, seq_a, seq_b)
    
    # Initialize DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m+1, n+1), dtype=int)
    
    # Initialize gap penalties
    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] + gap_open + (i-1)*gap_extend
    for j in range(1, n+1):
        dp[0][j] = dp[0][j-1] + gap_open + (j-1)*gap_extend
    
    # Fill DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Calculate scores for all possible operations
            match_score = dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            delete_score = dp[i-1][j] + gap_extend
            insert_score = dp[i][j-1] + gap_extend
            
            # Choose the maximum score
            dp[i][j] = max(match_score, delete_score, insert_score)
    
    # Find the optimal score
    if mode == "global":
        score = dp[m][n]
    else:
        # Find the maximum score in the DP table for local mode
        score = np.max(dp)
        # If no positive score found, return (0, "", "")
        if score <= 0:
            return (0, "", "")
        # Find the position of the maximum score
        for i in range(m, 0, -1):
            for j in range(n, 0, -1):
                if dp[i][j] == score:
                    # Create the aligned sequences
                    aligned_a = seq_a[:i] + "-" * (m - i)
                    aligned_b = seq_b[:j] + "-" * (n - j)
                    return (score, aligned_a, aligned_b)
    
    # Traceback to find the alignment
    i, j = m, n
    aligned_a = ""
    aligned_b = ""
    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq_a[i-1] == seq_b[j-1]:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + gap_extend:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + gap_extend:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
    
    return (score, aligned_a, aligned_b)