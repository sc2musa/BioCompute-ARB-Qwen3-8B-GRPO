import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Handle empty sequences
    if not seq_a or not seq_b:
        return (0, seq_a, seq_b)
    
    # Initialize DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m+1, n+1), dtype=int)
    
    # Initialize first row and column
    for i in range(m+1):
        dp[i, 0] = i * gap_open
    for j in range(n+1):
        dp[0, j] = j * gap_open
    
    # Fill DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Calculate score for match/mismatch
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Calculate gap scores
            gap_a = dp[i-1, j] + gap_extend
            gap_b = dp[i, j-1] + gap_extend
            
            # Choose maximum score
            dp[i, j] = max(score + dp[i-1, j-1], gap_a, gap_b)
    
    # Find the maximum score in the DP table
    max_score = np.max(dp)
    
    # Backtrack to find the optimal alignment
    i, j = np.unravel_index(max_score, dp.shape)
    aligned_a, aligned_b = [], []
    
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
    
    # Reverse the alignments
    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]
    
    # Return the result
    return (max_score, "".join(aligned_a), "".join(aligned_b))