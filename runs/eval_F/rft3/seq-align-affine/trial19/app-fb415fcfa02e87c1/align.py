import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    # Initialize the DP matrix
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m + 1, n + 1))
    back = np.zeros((m + 1, n + 1), dtype=int)

    # Initialize the first row and column
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] + gap_extend
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] + gap_extend

    # Fill the DP matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Calculate the score for match/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            # Calculate the score for gap extensions
            gap_score = gap_extend
            # Calculate the score for opening a new gap
            if i > 1 and j > 1:
                gap_score = max(gap_open, gap_extend)
            # Update the DP matrix
            dp[i][j] = max(
                dp[i-1][j-1] + score,
                dp[i-1][j] + gap_score,
                dp[i][j-1] + gap_score
            )
            # Update the backtracking matrix
            if dp[i][j] == dp[i-1][j-1] + score:
                back[i][j] = 0
            elif dp[i][j] == dp[i-1][j] + gap_score:
                back[i][j] = 1
            else:
                back[i][j] = 2

    # Find the optimal score
    if mode == "global":
        score = dp[m][n]
        aligned_a = ""
        aligned_b = ""
        i, j = m, n
        while i > 0 and j > 0:
            if back[i][j] == 0:
                aligned_a += seq_a[i-1]
                aligned_b += seq_b[j-1]
                i -= 1
                j -= 1
            elif back[i][j] == 1:
                aligned_a += "-"
                aligned_b += seq_b[j-1]
                j -= 1
            else:
                aligned_a += seq_a[i-1]
                aligned_b += "-"
                i -= 1
        aligned_a = aligned_a[::-1]
        aligned_b = aligned_b[::-1]
        return (score, aligned_a, aligned_b)
    else:
        # For local alignment, find the maximum score in the matrix
        score = np.max(dp)
        # If the maximum score is zero, return (0, "", "")
        if score == 0:
            return (0, "", "")
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        # Reconstruct the alignment
        aligned_a = ""
        aligned_b = ""
        while i > 0 and j > 0:
            if back[i][j] == 0:
                aligned_a += seq_a[i-1]
                aligned_b += seq_b[j-1]
                i -= 1
                j -= 1
            elif back[i][j] == 1:
                aligned_a += "-"
                aligned_b += seq_b[j-1]
                j -= 1
            else:
                aligned_a += seq_a[i-1]
                aligned_b += "-"
                i -= 1
        # Reverse the alignment
        aligned_a = aligned_a[::-1]
        aligned_b = aligned_b[::-1]
        return (score, aligned_a, aligned_b)