import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    # Initialize the DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    # Initialize the first row and column based on the mode
    if mode == "global":
        # Needleman-Wunsch initialization
        for i in range(m + 1):
            dp[i][0] = i * gap_open
        for j in range(n + 1):
            dp[0][j] = j * gap_open
    elif mode == "local":
        # Smith-Waterman initialization
        for i in range(m + 1):
            dp[i][0] = 0
        for j in range(n + 1):
            dp[0][j] = 0

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Score for match/mismatch
            match_score = dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            # Score for gap in seq_a
            gap_a_score = dp[i-1][j] + gap_open + gap_extend
            # Score for gap in seq_b
            gap_b_score = dp[i][j-1] + gap_open + gap_extend
            # Choose the maximum score
            dp[i][j] = max(match_score, gap_a_score, gap_b_score)

    # Traceback to find the optimal alignment
    i, j = m, n
    aligned_a, aligned_b = ",", ""
    while i > 0 and j > 0:
        if seq_a[i-1] == seq_b[j-1]:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        elif dp[i-1][j] == dp[i][j]:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
        elif dp[i][j-1] == dp[i][j]:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            j -= 1
        else:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1

    # Handle remaining characters
    while i > 0:
        aligned_a = seq_a[i-1] + aligned_a
        aligned_b = "-" + aligned_b
        i -= 1
    while j > 0:
        aligned_a = "-" + aligned_a
        aligned_b = seq_b[j-1] + aligned_b
        j -= 1

    # Return the score and aligned sequences
    score = dp[m][n]
    return (score, aligned_a, aligned_b)