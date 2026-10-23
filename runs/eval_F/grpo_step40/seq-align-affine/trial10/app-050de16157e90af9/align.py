import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Fill first row (gaps in seq_a)
    for j in range(1, len_b + 1):
        dp[0, j] = gap_open + (j - 1) * gap_extend
    
    # Fill first column (gaps in seq_b)
    for i in range(1, len_a + 1):
        dp[i, 0] = gap_open + (i - 1) * gap_extend
    
    # Fill the rest of the table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate score for match/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Calculate score for gap in seq_a
            score_a = dp[i-1, j] + gap_extend
            
            # Calculate score for gap in seq_b
            score_b = dp[i, j-1] + gap_extend
            
            # Choose the maximum score
            dp[i, j] = max(score, score_a, score_b)
    
    # Find the optimal score
    if mode == "global":
        score = dp[len_a, len_b]
    else:
        # For local, find the maximum value in the table
        score = np.max(dp)
        # Find the position of the maximum value
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        # Truncate the sequences to the matching region
        seq_a = seq_a[:i]
        seq_b = seq_b[:j]
        len_a, len_b = len(seq_a), len(seq_b)
        # Recalculate the DP table for the truncated sequences
        dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
        for j in range(1, len_b + 1):
            dp[0, j] = gap_open + (j - 1) * gap_extend
        for i in range(1, len_a + 1):
            dp[i, 0] = gap_open + (i - 1) * gap_extend
        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                if seq_a[i-1] == seq_b[j-1]:
                    score = match
                else:
                    score = mismatch
                score_a = dp[i-1, j] + gap_extend
                score_b = dp[i, j-1] + gap_extend
                dp[i, j] = max(score, score_a, score_b)
        score = dp[len_a, len_b]
    
    # Trace back to find the alignment
    aligned_a = ""
    aligned_b = ""
    i, j = len_a, len_b
    while i > 0 and j > 0:
        if seq_a[i-1] == seq_b[j-1]:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        elif dp[i, j] == dp[i-1, j] + gap_extend:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
        else:
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
    
    return (score, aligned_a, aligned_b)