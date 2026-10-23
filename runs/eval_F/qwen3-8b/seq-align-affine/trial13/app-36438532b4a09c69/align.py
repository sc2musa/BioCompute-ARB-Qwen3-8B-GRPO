import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    m, n = len(seq_a), len(seq_b)

    # Initialize the DP table
    dp = np.zeros((m + 1, n + 1), dtype=int)

    # Initialize the first row and column
    for i in range(m + 1):
        dp[i][0] = i * gap_open + (i - 1) * gap_extend
    for j in range(n + 1):
        dp[0][j] = j * gap_open + (j - 1) * gap_extend

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Match/mismatch score
            match_score = dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            # Gap in seq_a
            gap_a_score = dp[i-1][j] + gap_open + (i - 1) * gap_extend
            # Gap in seq_b
            gap_b_score = dp[i][j-1] + gap_open + (j - 1) * gap_extend
            # Choose the maximum score
            dp[i][j] = max(match_score, gap_a_score, gap_b_score)

    # Extract the score
    score = dp[m][n]

    # Backtrack to find the alignment
    aligned_a, aligned_b = ",", ","
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + gap_open + (i - 1) * gap_extend:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + gap_open + (j - 1) * gap_extend:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
        else:
            # This should not happen for global alignment
            break

    return (score, aligned_a, aligned_b)