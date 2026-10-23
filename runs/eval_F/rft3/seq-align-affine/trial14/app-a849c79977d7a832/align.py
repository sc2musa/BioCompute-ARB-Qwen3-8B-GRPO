import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m+1, n+1))
    
    # Initialize first row with gap penalties
    for i in range(1, m+1):
        dp[i][0] = gap_open + (i-1)*gap_extend
    
    # Initialize first column with gap penalties
    for j in range(1, n+1):
        dp[0][j] = gap_open + (j-1)*gap_extend
    
    # Fill DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Calculate score for match/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Calculate score for gap in seq_a
            gap_a = dp[i-1][j] + gap_extend
            # Calculate score for gap in seq_b
            gap_b = dp[i][j-1] + gap_extend
            
            # Choose maximum score
            dp[i][j] = max(score, gap_a, gap_b)
    
    # Traceback to find optimal alignment
    i, j = m, n
    aligned_a = []
    aligned_b = []
    
    while i > 0 and j > 0:
        if dp[i][j] == dp[i-1][j] + gap_extend:
            # Gap in seq_a
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            i -= 1
        elif dp[i][j] == dp[i][j-1] + gap_extend:
            # Gap in seq_b
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            j -= 1
        else:
            # Match/mismatch
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
    
    # Reverse to get correct order
    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]
    
    # Return score and aligned sequences
    return (dp[m][n], "".join(aligned_a), "".join(aligned_b))