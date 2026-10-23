import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m+1, n+1), dtype=int)
    
    # Precompute gap penalties
    gap_open = gap_open
    gap_extend = gap_extend
    
    # Fill the DP table
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 and j == 0:
                dp[i, j] = 0
            elif i == 0:
                # Gap in seq_a
                dp[i, j] = dp[i, j-1] + gap_open + (j-1)*gap_extend
            elif j == 0:
                # Gap in seq_b
                dp[i, j] = dp[i-1, j] + gap_open + (i-1)*gap_extend
            else:
                # Match or mismatch
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                dp[i, j] = max(
                    dp[i-1, j-1] + score,
                    dp[i-1, j] + gap_open + (i-1)*gap_extend,
                    dp[i, j-1] + gap_open + (j-1)*gap_extend
                )
    
    # Extract the score
    score = dp[m, n]
    
    # Backtrack to find the alignment
    aligned_a = ["-" for _ in range(m+n)]
    aligned_b = ["-" for _ in range(m+n)]
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq_a[i-1] == seq_b[j-1]:
            aligned_a[i-1] = seq_a[i-1]
            aligned_b[j-1] = seq_b[j-1]
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i-1, j] + gap_open + (i-1)*gap_extend:
            aligned_a[i-1] = seq_a[i-1]
            aligned_b[j-1] = "-"
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j-1] + gap_open + (j-1)*gap_extend:
            aligned_a[i-1] = "-"
            aligned_b[j-1] = seq_b[j-1]
            j -= 1
    
    # Return the result
    return (score, "".join(aligned_a), "".join(aligned_b))