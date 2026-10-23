import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Handle empty sequences
    if not seq_a or not seq_b:
        return (0, seq_a, seq_b)
    
    # Initialize DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a+1, len_b+1), dtype=int)
    
    # Fill DP table
    for i in range(len_a+1):
        for j in range(len_b+1):
            if i == 0 and j == 0:
                # Starting point
                dp[i, j] = 0
            elif i == 0:
                # Only gaps in seq_b
                dp[i, j] = dp[i, j-1] + gap_extend
            elif j == 0:
                # Only gaps in seq_a
                dp[i, j] = dp[i-1, j] + gap_extend
            else:
                # Calculate score for match/mismatch
                if seq_a[i-1] == seq_b[j-1]:
                    score = match
                else:
                    score = mismatch
                
                # Calculate gap penalties
                gap_a = gap_open if i > 1 and dp[i-1, j] == dp[i-2, j] else gap_extend
                gap_b = gap_open if j > 1 and dp[i, j-1] == dp[i, j-2] else gap_extend
                
                # Choose maximum score
                dp[i, j] = max(
                    dp[i-1, j] + gap_a,
                    dp[i, j-1] + gap_b,
                    dp[i-1, j-1] + score
                )
    
    # Find optimal alignment
    i, j = len_a, len_b
    aligned_a = ["" for _ in range(len_a + len_b)]
    aligned_b = ["" for _ in range(len_a + len_b)]
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a[i-1] = seq_a[i-1]
                aligned_b[j-1] = seq_b[j-1]
                i -= 1
                j -= 1
            else:
                # Choose the best path
                if dp[i-1, j] + gap_open == dp[i, j]:
                    aligned_a[i-1] = "-"
                    aligned_b[j-1] = seq_b[j-1]
                    i -= 1
                elif dp[i, j-1] + gap_open == dp[i, j]:
                    aligned_a[i-1] = seq_a[i-1]
                    aligned_b[j-1] = "-"
                    j -= 1
                else:
                    aligned_a[i-1] = "-"
                    aligned_b[j-1] = "-"
                    i -= 1
                    j -= 1
        elif i > 0:
            aligned_a[i-1] = "-"
            aligned_b[j-1] = "-"
            i -= 1
        else:
            aligned_a[i-1] = "-"
            aligned_b[j-1] = "-"
            j -= 1
    
    # Return the result
    return (dp[len_a, len_b], ''.join(aligned_a), ''.join(aligned_b))