import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize DP table
    len_a = len(seq_a)
    len_b = len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=np.int32)
    
    # Precompute score matrix
    score = np.zeros((len_a, len_b), dtype=np.int32)
    for i in range(len_a):
        for j in range(len_b):
            score[i, j] = match if seq_a[i] == seq_b[j] else mismatch
    
    # Initialize first row and column
    for i in range(1, len_a + 1):
        dp[i, 0] = dp[i-1, 0] + gap_open + (i-1)*gap_extend
    for j in range(1, len_b + 1):
        dp[0, j] = dp[0, j-1] + gap_open + (j-1)*gap_extend
    
    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            dp[i, j] = max(
                score[i-1, j-1] + dp[i-1, j-1],
                dp[i-1, j] + gap_extend,
                dp[i, j-1] + gap_extend
            )
    
    # Handle local mode
    if mode == "local":
        max_score = np.max(dp)
        # Find the position of the maximum score
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if dp[i, j] == max_score:
                    # Trace back from this position
                    aligned_a = ""
                    aligned_b = ""
                    i2, j2 = i, j
                    while i2 > 0 or j2 > 0:
                        if i2 > 0 and j2 > 0 and dp[i2, j2] == dp[i2-1, j2-1] + score[i2-1, j2-1]:
                            aligned_a = seq_a[i2-1] + aligned_a
                            aligned_b = seq_b[j2-1] + aligned_b
                            i2 -= 1
                            j2 -= 1
                        elif i2 > 0 and dp[i2, j2] == dp[i2-1, j2] + gap_extend:
                            aligned_a = seq_a[i2-1] + aligned_a
                            aligned_b = "-" + aligned_b
                            i2 -= 1
                        elif j2 > 0 and dp[i2, j2] == dp[i2, j2-1] + gap_extend:
                            aligned_a = "-" + aligned_a
                            aligned_b = seq_b[j2-1] + aligned_b
                            j2 -= 1
                    return (max_score, aligned_a, aligned_b)
    
    # Handle global mode
    i, j = len_a, len_b
    aligned_a = ""
    aligned_b = ""
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i, j] == dp[i-1, j-1] + score[i-1, j-1]:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i-1, j] + gap_extend:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j-1] + gap_extend:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
    
    return (dp[len_a, len_b], aligned_a, aligned_b)