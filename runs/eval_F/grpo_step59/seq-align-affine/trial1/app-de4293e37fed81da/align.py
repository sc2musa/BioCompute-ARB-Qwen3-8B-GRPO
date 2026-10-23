import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize DP table
    len_a = len(seq_a)
    len_b = len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize first row and column with gap penalties
    for i in range(1, len_a + 1):
        dp[i, 0] = dp[i-1, 0] + gap_open + (i-1) * gap_extend
    for j in range(1, len_b + 1):
        dp[0, j] = dp[0, j-1] + gap_open + (j-1) * gap_extend
    
    # Fill the DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate scores for different operations
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            gap_a = dp[i-1, j] + gap_extend
            gap_b = dp[i, j-1] + gap_extend
            
            # Choose the maximum score
            dp[i, j] = max(match_score, gap_a, gap_b)
    
    # Get the maximum score
    if mode == "global":
        max_score = dp[len_a, len_b]
    else:
        max_score = np.max(dp)
    
    # Traceback to find the alignment
    aligned_a = ""
    aligned_b = ""
    i = len_a
    j = len_b
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            # Check if we came from a match
            if dp[i, j] == dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a += seq_a[i-1]
                aligned_b += seq_b[j-1]
                i -= 1
                j -= 1
            # Check if we came from a gap in seq_a
            elif dp[i, j] == dp[i-1, j] + gap_extend:
                aligned_a += seq_a[i-1]
                aligned_b += "-"
                i -= 1
            # Check if we came from a gap in seq_b
            else:
                aligned_a += "-"
                aligned_b += seq_b[j-1]
                j -= 1
        elif i > 0:
            aligned_a += seq_a[i-1]
            aligned_b += "-"
            i -= 1
        else:
            aligned_a += "-"
            aligned_b += seq_b[j-1]
            j -= 1
    
    # Reverse the alignment
    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]
    
    return (max_score, aligned_a, aligned_b)