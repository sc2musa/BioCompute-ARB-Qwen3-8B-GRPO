import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize first row and column
    for i in range(1, len_a + 1):
        dp[i][0] = dp[i-1][0] + gap_open + (gap_extend * (i-1))
    for j in range(1, len_b + 1):
        dp[0][j] = dp[0][j-1] + gap_open + (gap_extend * (j-1))
    
    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate score for matching characters
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Calculate max score
            dp[i][j] = max(
                dp[i-1][j-1] + score,
                dp[i-1][j] + gap_open + (gap_extend * (i-1)),
                dp[i][j-1] + gap_open + (gap_extend * (j-1))
            )
    
    # Trace back to find alignment
    aligned_a = ["-" for _ in range(len_a)]
    aligned_b = ["-" for _ in range(len_b)]
    i, j = len_a, len_b
    while i > 0 and j > 0:
        if seq_a[i-1] == seq_b[j-1]:
            aligned_a[i-1] = seq_a[i-1]
            aligned_b[j-1] = seq_b[j-1]
            i -= 1
            j -= 1
        elif dp[i-1][j] == dp[i][j]:
            aligned_a[i-1] = "-"
            aligned_b[j-1] = seq_b[j-1]
            i -= 1
        else:
            aligned_a[i-1] = seq_a[i-1]
            aligned_b[j-1] = "-"
            j -= 1
    
    # Return result
    return (dp[len_a][len_b], "".join(aligned_a), "".join(aligned_b))