import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    # Implement the align function here
    # Using dynamic programming for pairwise alignment
    # Initialize the DP table
    len_a = len(seq_a)
    len_b = len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1))
    # Fill the DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate the score for aligning the current characters
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            # Calculate the score for inserting a gap in seq_a
            score_gap_a = dp[i-1, j] + gap_extend
            # Calculate the score for inserting a gap in seq_b
            score_gap_b = dp[i, j-1] + gap_extend
            # Calculate the score for aligning the current characters
            score_align = dp[i-1, j-1] + score
            # Choose the maximum score
            dp[i, j] = max(score_align, score_gap_a, score_gap_b)
    # Find the optimal alignment
    if mode == "global":
        # For global alignment, the score is the value at the bottom-right corner
        score = dp[len_a, len_b]
        # Backtrack to find the aligned sequences
        aligned_a = ""
        aligned_b = ""
        i = len_a
        j = len_b
        while i > 0 and j > 0:
            if dp[i, j] == dp[i-1, j-1] + score:
                # Align the current characters
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif dp[i, j] == dp[i-1, j] + gap_extend:
                # Insert a gap in seq_a
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
            else:
                # Insert a gap in seq_b
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
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
    else:
        # For local alignment, find the maximum score and backtrack
        max_score = 0
        max_i = 0
        max_j = 0
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if dp[i, j] > max_score:
                    max_score = dp[i, j]
                    max_i = i
                    max_j = j
        # Backtrack to find the aligned sequences
        aligned_a = ""
        aligned_b = ""
        i = max_i
        j = max_j
        while i > 0 and j > 0:
            if dp[i, j] == dp[i-1, j-1] + score:
                # Align the current characters
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif dp[i, j] == dp[i-1, j] + gap_extend:
                # Insert a gap in seq_a
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
            else:
                # Insert a gap in seq_b
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
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
    return (score, aligned_a, aligned_b)