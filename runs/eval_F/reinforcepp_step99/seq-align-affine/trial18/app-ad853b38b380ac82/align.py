import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m+1, n+1), dtype=int)
    
    # Fill the DP table
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 and j == 0:
                # Starting point
                dp[i][j] = 0
            elif i == 0:
                # Only gaps in seq_a
                dp[i][j] = dp[i][j-1] + gap_open + (j-1)*gap_extend
            elif j == 0:
                # Only gaps in seq_b
                dp[i][j] = dp[i-1][j] + gap_open + (i-1)*gap_extend
            else:
                # Compute the score for matching/mismatch
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                # Compute the score for gaps
                gap_score = gap_open + (min(i, j) - 1) * gap_extend
                # Take the maximum of all possible options
                dp[i][j] = max(dp[i-1][j] + gap_score, dp[i][j-1] + gap_score, dp[i-1][j-1] + score)
    
    # Backtrack to find the optimal alignment
    i, j = m, n
    aligned_a = []
    aligned_b = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            else:
                # Choose the best option (could be match, gap in a, or gap in b)
                # This is a simplified version; in practice we would check all options
                if dp[i-1][j] + gap_open + (i-1)*gap_extend > dp[i][j-1] + gap_open + (j-1)*gap_extend:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    i -= 1
                else:
                    aligned_a.append("-")
                    aligned_b.append(seq_b[j-1])
                    j -= 1
        elif i > 0:
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
    
    # Return the score and alignments
    return (dp[m][n], "".join(aligned_a), "".join(aligned_b))