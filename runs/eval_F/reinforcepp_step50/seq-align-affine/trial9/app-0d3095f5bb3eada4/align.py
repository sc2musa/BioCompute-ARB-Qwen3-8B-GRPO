import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize first row and column with gap penalties
    for i in range(1, len_a + 1):
        dp[i, 0] = gap_open + (i - 1) * gap_extend
    for j in range(1, len_b + 1):
        dp[0, j] = gap_open + (j - 1) * gap_extend
    
    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate score for match/mismatch
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Calculate score for gap in seq_a
            gap_a = dp[i-1, j] + gap_extend
            
            # Calculate score for gap in seq_b
            gap_b = dp[i, j-1] + gap_extend
            
            # Choose the maximum score
            dp[i, j] = max(score + dp[i-1, j-1], gap_a, gap_b)
    
    # Traceback to find the optimal alignment
    i, j = len_a, len_b
    aligned_a, aligned_b = "", ""
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            # Check if this is a match/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            else:
                # Check if this is a gap in seq_a or seq_b
                if dp[i-1, j] + gap_extend == dp[i, j]:
                    aligned_a = "-" + aligned_a
                    aligned_b = seq_b[j-1] + aligned_b
                    i -= 1
                elif dp[i, j-1] + gap_extend == dp[i, j]:
                    aligned_a = seq_a[i-1] + aligned_a
                    aligned_b = "-" + aligned_b
                    j -= 1
                else:
                    # This should not happen
                    return (0, "", "")
        elif i > 0:
            # Gap in seq_a
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
        else:
            # Gap in seq_b
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            j -= 1
    
    # Return the score and aligned sequences
    return (dp[len_a, len_b], aligned_a, aligned_b)