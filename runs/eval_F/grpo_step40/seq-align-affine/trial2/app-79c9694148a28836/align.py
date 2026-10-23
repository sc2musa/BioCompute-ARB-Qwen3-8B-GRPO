import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m+1, n+1))
    
    # Initialize the first row and column
    for i in range(m+1):
        dp[i, 0] = i * gap_open + (i-1) * gap_extend if i > 0 else 0
    for j in range(n+1):
        dp[0, j] = j * gap_open + (j-1) * gap_extend if j > 0 else 0
    
    # Fill the DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Calculate the score for matching characters
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Calculate the score for gaps
            gap_a = dp[i-1, j] + gap_extend
            gap_b = dp[i, j-1] + gap_extend
            
            # Choose the maximum score
            dp[i, j] = max(match_score + dp[i-1, j-1], gap_a, gap_b)
    
    # Find the optimal alignment
    if mode == "global":
        score = dp[m, n]
        i, j = m, n
    else:
        # Find the maximum score in the DP table
        max_score = np.max(dp)
        # Find the position of the maximum score
        for i in range(m+1):
            for j in range(n+1):
                if dp[i, j] == max_score:
                    break
            else:
                continue
            break
        score = max_score
        i, j = m, n
    
    # Backtrack to find the aligned sequences
    aligned_a = []
    aligned_b = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i, j] == dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i-1, j] + gap_extend:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        else:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
    
    # Reverse the aligned sequences
    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]
    
    return (score, "".join(aligned_a), "".join(aligned_b))