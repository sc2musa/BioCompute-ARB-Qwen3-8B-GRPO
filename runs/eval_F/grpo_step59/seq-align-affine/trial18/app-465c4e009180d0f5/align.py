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
        dp[i][0] = dp[i-1][0] + gap_open + (i-1) * gap_extend
    for j in range(1, len_b + 1):
        dp[0][j] = dp[0][j-1] + gap_open + (j-1) * gap_extend
    
    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Match/mismatch
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            # Calculate possible scores
            score1 = dp[i-1][j-1] + match_score
            score2 = dp[i-1][j] + gap_extend
            score3 = dp[i][j-1] + gap_extend
            # Choose the maximum score
            dp[i][j] = max(score1, score2, score3)
    
    # Find optimal score
    if mode == "global":
        score = dp[len_a][len_b]
    else:
        # Find maximum score in the table
        score = np.max(dp)
        # Find the position of the maximum score
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if dp[i][j] == score:
                    # Create aligned sequences
                    aligned_a = "".join([seq_a[k] if k < i else "-" for k in range(len_a)])
                    aligned_b = "".join([seq_b[k] if k < j else "-" for k in range(len_b)])
                    return (score, aligned_a, aligned_b)
    
    # Create aligned sequences for global mode
    aligned_a = "".join([seq_a[k] if k < len_a else "-" for k in range(len_a)])
    aligned_b = "".join([seq_b[k] if k < len_b else "-" for k in range(len_b)])
    return (score, aligned_a, aligned_b)