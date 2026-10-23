import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Fill the DP table
    for i in range(len_a + 1):
        for j in range(len_b + 1):
            if i == 0 and j == 0:
                dp[i, j] = 0
            elif i == 0:
                dp[i, j] = dp[i, j-1] + gap_open
            elif j == 0:
                dp[i, j] = dp[i-1, j] + gap_open
            else:
                # Calculate the score for matching/mismatch
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                # Calculate the score for gap penalties
                gap_score = gap_open + (i + j - 2) * gap_extend
                # Choose the maximum score
                dp[i, j] = max(dp[i-1, j] + gap_open, dp[i, j-1] + gap_open, dp[i-1, j-1] + score)
    
    # Extract the optimal score
    score = dp[len_a, len_b]
    
    # Backtrack to find the alignment
    aligned_a = ["-" * len_b] * len_a
    aligned_b = ["-" * len_a] * len_b
    i, j = len_a, len_b
    while i > 0 and j > 0:
        if seq_a[i-1] == seq_b[j-1]:
            aligned_a[i-1] = seq_a[i-1]
            aligned_b[j-1] = seq_b[j-1]
            i -= 1
            j -= 1
        elif dp[i-1, j] + gap_open == dp[i, j]:
            aligned_a[i-1] = "-"
            aligned_b[j-1] = seq_b[j-1]
            i -= 1
        elif dp[i, j-1] + gap_open == dp[i, j]:
            aligned_a[i-1] = seq_a[i-1]
            aligned_b[j-1] = "-"
            j -= 1
        else:
            # This should not happen with the correct implementation
            aligned_a[i-1] = "-"
            aligned_b[j-1] = "-"
            i -= 1
            j -= 1
    
    # Handle remaining characters
    while i > 0:
        aligned_a[i-1] = "-"
        i -= 1
    while j > 0:
        aligned_b[j-1] = "-"
        j -= 1
    
    # Return the result
    return (score, "".join(aligned_a), "".join(aligned_b))