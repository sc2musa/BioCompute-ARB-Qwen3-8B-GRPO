import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Base cases for gaps
    for i in range(1, len_a + 1):
        dp[i][0] = gap_open + (i - 1) * gap_extend
    for j in range(1, len_b + 1):
        dp[0][j] = gap_open + (j - 1) * gap_extend
    
    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate scores for all possible operations
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            gap_a = dp[i-1][j] + gap_extend
            gap_b = dp[i][j-1] + gap_extend
            
            # Choose the maximum score
            dp[i][j] = max(match_score + dp[i-1][j-1], gap_a, gap_b)
    
    # Find the optimal score
    if mode == "global":
        score = dp[len_a][len_b]
    else:
        # For local mode, find the maximum score in the table
        score = np.max(dp)
        # If the score is zero and all are zero, return (0, "", "")
        if score == 0:
            return (0, "", "")
        # Otherwise, backtrack to find the alignment
        aligned_a = ""
        aligned_b = ""
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        while i > 0 and j > 0:
            if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a += seq_a[i-1]
                aligned_b += seq_b[j-1]
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i-1][j] + gap_extend:
                aligned_a += seq_a[i-1]
                aligned_b += "-"
                i -= 1
            else:
                aligned_a += "-"
                aligned_b += seq_b[j-1]
                j -= 1
        # Reverse the aligned sequences
        aligned_a = aligned_a[::-1]
        aligned_b = aligned_b[::-1]
        
    return (score, aligned_a, aligned_b)