import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    # Implementation goes here
    # Create DP tables
    len_a, len_b = len(seq_a), len(seq_b)
    if mode == "global":
        # Global alignment (Needleman-Wunsch)
        # Initialize DP table
        dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
        # Fill first row and column
        for i in range(1, len_a + 1):
            dp[i, 0] = dp[i-1, 0] + gap_open
        for j in range(1, len_b + 1):
            dp[0, j] = dp[0, j-1] + gap_open
        # Fill DP table
        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                # Calculate score for match/mismatch
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                # Calculate score for gap in seq_a
                gap_a = dp[i-1, j] + gap_extend
                # Calculate score for gap in seq_b
                gap_b = dp[i, j-1] + gap_extend
                # Choose the best score
                dp[i, j] = max(score, gap_a, gap_b)
        # Get the maximum score
        max_score = dp[len_a, len_b]
        # Backtrack to find the alignment
        aligned_a, aligned_b = "", ""
        i, j = len_a, len_b
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif dp[i-1, j] == dp[i, j]:
                # Gap in seq_a
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
            else:
                # Gap in seq_b
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                j -= 1
        return (max_score, aligned_a, aligned_b)
    elif mode == "local":
        # Local alignment (Smith-Waterman)
        # Initialize DP table
        dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
        # Fill DP table
        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                # Calculate score for match/mismatch
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                # Calculate score for gap in seq_a
                gap_a = dp[i-1, j] + gap_extend
                # Calculate score for gap in seq_b
                gap_b = dp[i, j-1] + gap_extend
                # Choose the best score
                dp[i, j] = max(score, gap_a, gap_b, 0)
        # Find the maximum score
        max_score = np.max(dp)
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        # Backtrack to find the alignment
        aligned_a, aligned_b = "", ""
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif dp[i-1, j] == dp[i, j]:
                # Gap in seq_a
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
            else:
                # Gap in seq_b
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                j -= 1
        return (max_score, aligned_a, aligned_b)
    else:
        raise ValueError("Invalid mode: must be 'global' or 'local'")