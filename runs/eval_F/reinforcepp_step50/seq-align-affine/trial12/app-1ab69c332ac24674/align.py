import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the DP table
    len_a = len(seq_a)
    len_b = len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize the first row and column
    for i in range(1, len_a + 1):
        dp[i, 0] = gap_open + (i - 1) * gap_extend
    for j in range(1, len_b + 1):
        dp[0, j] = gap_open + (j - 1) * gap_extend
    
    # Fill the DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate the score for matching/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Calculate the score for gaps
            # Corrected gap score calculation
            gap_score = gap_open + (i - 1) * gap_extend
            gap_score += (j - 1) * gap_extend
            
            # Calculate the maximum score
            dp[i, j] = max(
                dp[i-1, j-1] + score,
                dp[i-1, j] + gap_extend,
                dp[i, j-1] + gap_extend,
                gap_score
            )
    
    # Find the optimal score
    if mode == "global":
        score = dp[len_a, len_b]
        # Traceback to find the alignment
        aligned_a = ""
        aligned_b = ""
        i, j = len_a, len_b
        while i > 0 and j > 0:
            if dp[i, j] == dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a += seq_a[i-1]
                aligned_b += seq_b[j-1]
                i -= 1
                j -= 1
            elif dp[i, j] == dp[i-1, j] + gap_extend:
                aligned_a += seq_a[i-1]
                aligned_b += "-"
                i -= 1
            else:
                aligned_a += "-"
                aligned_b += seq_b[j-1]
                j -= 1
        # Add any remaining characters
        while i > 0:
            aligned_a += seq_a[i-1]
            aligned_b += "-"
            i -= 1
        while j > 0:
            aligned_a += "-"
            aligned_b += seq_b[j-1]
            j -= 1
        # Reverse the alignments
        aligned_a = aligned_a[::-1]
        aligned_b = aligned_b[::-1]
    else:
        # For local mode, find the maximum score in the table
        score = np.max(dp)
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        # Traceback to find the alignment
        aligned_a = ""
        aligned_b = ""
        while i > 0 and j > 0:
            if dp[i, j] == dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a += seq_a[i-1]
                aligned_b += seq_b[j-1]
                i -= 1
                j -= 1
            elif dp[i, j] == dp[i-1, j] + gap_extend:
                aligned_a += seq_a[i-1]
                aligned_b += "-"
                i -= 1
            else:
                aligned_a += "-"
                aligned_b += seq_b[j-1]
                j -= 1
        # Add any remaining characters
        while i > 0:
            aligned_a += seq_a[i-1]
            aligned_b += "-"
            i -= 1
        while j > 0:
            aligned_a += "-"
            aligned_b += seq_b[j-1]
            j -= 1
        # Reverse the alignments
        aligned_a = aligned_a[::-1]
        aligned_b = aligned_b[::-1]
    
    return (score, aligned_a, aligned_b)