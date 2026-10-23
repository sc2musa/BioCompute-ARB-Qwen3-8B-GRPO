import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Handle empty sequences
    if not seq_a and not seq_b:
        return (0, "", "")
    if not seq_a:
        return (0, "", seq_b)
    if not seq_b:
        return (0, seq_a, "")
    
    # Initialize DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    # Initialize first row and column
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] + gap_open + (i-1)*gap_extend
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] + gap_open + (j-1)*gap_extend
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Match/mismatch
            match_score = dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            
            # Gap in seq_a
            gap_a_score = dp[i-1][j] + gap_open + (i-1)*gap_extend
            
            # Gap in seq_b
            gap_b_score = dp[i][j-1] + gap_open + (j-1)*gap_extend
            
            # Choose the maximum score
            dp[i][j] = max(match_score, gap_a_score, gap_b_score)
    
    # Backtrack to find the alignment
    i, j = m, n
    aligned_a, aligned_b = "", ""
    
    while i > 0 and j > 0:
        if seq_a[i-1] == seq_b[j-1]:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        else:
            # Check if we came from a match/mismatch
            if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            # Check if we came from a gap in seq_a
            elif dp[i][j] == dp[i-1][j] + gap_open + (i-1)*gap_extend:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
            # Check if we came from a gap in seq_b
            else:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                j -= 1
    
    # Handle remaining characters
    while i > 0:
        aligned_a = "-" + aligned_a
        aligned_b = seq_b[j-1] + aligned_b
        i -= 1
        j -= 1
    while j > 0:
        aligned_a = seq_a[i-1] + aligned_a
        aligned_b = "-" + aligned_b
        i -= 1
        j -= 1
    
    return (dp[m][n], aligned_a, aligned_b)