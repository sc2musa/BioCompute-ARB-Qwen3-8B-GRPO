import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a = len(seq_a)
    len_b = len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)

    # Initialize first row and column
    for i in range(1, len_a + 1):
        dp[i, 0] = i * gap_open
    for j in range(1, len_b + 1):
        dp[0, j] = j * gap_open

    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Match/mismatch
            match_score = dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            # Gap in seq_a
            gap_a_score = dp[i-1, j] + gap_open
            # Gap in seq_b
            gap_b_score = dp[i, j-1] + gap_open
            dp[i, j] = max(match_score, gap_a_score, gap_b_score)

    # Backtrack to find aligned sequences
    aligned_a = ""
    aligned_b = ""
    i, j = len_a, len_b
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i, j] == dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a += seq_a[i-1]
            aligned_b += seq_b[j-1]
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i-1, j] + gap_open:
            aligned_a += seq_a[i-1]
            aligned_b += "-"
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j-1] + gap_open:
            aligned_a += "-"
            aligned_b += seq_b[j-1]
            j -= 1
        else:
            # Handle cases where all options are not valid
            pass

    return (dp[len_a, len_b], aligned_a[::-1], aligned_b[::-1])