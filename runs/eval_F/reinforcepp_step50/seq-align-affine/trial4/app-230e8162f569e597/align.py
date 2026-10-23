import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Handle empty sequences
    if not seq_a or not seq_b:
        return (0, seq_a, seq_b)
    
    # Initialize DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize first row and column
    for i in range(len_a + 1):
        dp[i, 0] = i * gap_open + (i - 1) * gap_extend if i > 0 else 0
    for j in range(len_b + 1):
        dp[0, j] = j * gap_open + (j - 1) * gap_extend if j > 0 else 0
    
    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate scores for match/mismatch
            match_score = dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            
            # Calculate scores for gaps
            gap_a = dp[i-1, j] + gap_extend if i > 0 else gap_open
            gap_b = dp[i, j-1] + gap_extend if j > 0 else gap_open
            
            # Choose the best score
            dp[i, j] = max(match_score, gap_a, gap_b)
    
    # Extract the optimal score
    score = dp[len_a, len_b]
    
    # Backtrack to find the alignment
    aligned_a = ["-" for _ in range(len_a + len_b)]
    aligned_b = ["-" for _ in range(len_a + len_b)]
    i, j = len_a, len_b
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq_a[i-1] == seq_b[j-1]:
            aligned_a[i-1] = seq_a[i-1]
            aligned_b[j-1] = seq_b[j-1]
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i-1, j] + gap_open:
            aligned_a[i-1] = seq_a[i-1]
            aligned_b[j-1] = "-"
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j-1] + gap_open:
            aligned_a[i-1] = "-"
            aligned_b[j-1] = seq_b[j-1]
            j -= 1
        else:
            # Handle gap extension
            if i > 0 and dp[i, j] == dp[i-1, j] + gap_extend:
                aligned_a[i-1] = seq_a[i-1]
                aligned_b[j-1] = "-"
                i -= 1
            elif j > 0 and dp[i, j] == dp[i, j-1] + gap_extend:
                aligned_a[i-1] = "-"
                aligned_b[j-1] = seq_b[j-1]
                j -= 1
            else:
                # This should never happen with proper initialization
                aligned_a[i-1] = "-"
                aligned_b[j-1] = "-"
                i -= 1
                j -= 1
    
    # Return the result
    return (score, "".join(aligned_a), "".join(aligned_b))