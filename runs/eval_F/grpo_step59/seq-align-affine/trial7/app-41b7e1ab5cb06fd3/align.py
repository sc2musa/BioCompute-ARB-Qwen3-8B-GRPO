import numpy as np


def align(seq_a, seq_b, mode="global",
          match=2, mismatch=-3,
          gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    # Handle empty sequences
    if not seq_a or not seq_b:
        return (0, seq_a, seq_b)

    # Initialize the DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    # Fill the first row and column with gap penalties
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] + gap_open if i == 1 else dp[i-1][0] + gap_extend
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] + gap_open if j == 1 else dp[0][j-1] + gap_extend

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Calculate the score for matching/mismatch
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            # Calculate the score for gap extensions
            gap_score = gap_extend
            # Calculate the score for opening a new gap
            if i > 1 and j > 1:
                gap_score = gap_open
            # Update the DP table
            dp[i][j] = max(
                dp[i-1][j-1] + score,
                dp[i-1][j] + gap_score,
                dp[i][j-1] + gap_score
            )

    # Backtrack to find the optimal alignment
    i, j = m, n
    aligned_a, aligned_b = [], []
    while i > 0 and j > 0:
        if seq_a[i-1] == seq_b[j-1]:
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] == dp[i][j]:
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            i -= 1
        elif dp[i][j-1] == dp[i][j]:
            aligned_a.append(seq_a[i-1])
            aligned_b.append('-')
            j -= 1
        else:
            # This case should not happen with the correct implementation
            aligned_a.append('-')
            aligned_b.append('-')
            i -= 1
            j -= 1

    # Handle remaining characters
    while i > 0:
        aligned_a.append(seq_a[i-1])
        aligned_b.append('-')
        i -= 1
    while j > 0:
        aligned_a.append('-')
        aligned_b.append(seq_b[j-1])
        j -= 1

    # Reverse the aligned sequences
    aligned_a = ''.join(reversed(aligned_a))
    aligned_b = ''.join(reversed(aligned_b))

    # Return the score and aligned sequences
    return (dp[m][n], aligned_a, aligned_b)