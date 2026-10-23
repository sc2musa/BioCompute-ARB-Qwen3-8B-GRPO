import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    # Initialize the first row and column
    for i in range(m + 1):
        dp[i, 0] = i * gap_open + (i - 1) * gap_extend
    for j in range(n + 1):
        dp[0, j] = j * gap_open + (j - 1) * gap_extend
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Calculate the score for matching/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Calculate the score for gap in seq_a
            gap_a = gap_open + (i - 1) * gap_extend
            # Calculate the score for gap in seq_b
            gap_b = gap_open + (j - 1) * gap_extend
            
            # Choose the maximum score
            dp[i, j] = max(
                dp[i-1, j-1] + score,
                dp[i-1, j] + gap_a,
                dp[i, j-1] + gap_b
            )
    
    # Extract the score
    score = dp[m, n]
    
    # Backtrack to find the alignment
    aligned_a = ""
    aligned_b = ""
    i, j = m, n
    while i > 0 and j > 0:
        if seq_a[i-1] == seq_b[j-1]:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        else:
            # Check if we should take a gap in seq_a or seq_b
            if dp[i-1, j] + gap_a == dp[i, j]:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
            elif dp[i, j-1] + gap_b == dp[i, j]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                j -= 1
            else:
                # This should not happen with the correct implementation
                aligned_a = "-" + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
                j -= 1
    
    # Handle remaining gaps
    while i > 0:
        aligned_a = "-" + aligned_a
        aligned_b = seq_b[j-1] + aligned_b
        i -= 1
    while j > 0:
        aligned_a = seq_a[i-1] + aligned_a
        aligned_b = "-" + aligned_b
        j -= 1
    
    return (score, aligned_a, aligned_b)