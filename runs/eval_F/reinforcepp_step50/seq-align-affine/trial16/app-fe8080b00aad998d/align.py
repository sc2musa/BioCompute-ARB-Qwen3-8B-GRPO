import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize first row and column
    for i in range(len_a + 1):
        dp[i, 0] = i * gap_open
    for j in range(len_b + 1):
        dp[0, j] = j * gap_open
    
    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate scores for different operations
            match_score = dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            gap_a_score = dp[i-1, j] + gap_extend
            gap_b_score = dp[i, j-1] + gap_extend
            
            # Choose the best score
            dp[i, j] = max(match_score, gap_a_score, gap_b_score)
    
    # Handle local mode
    if mode == "local":
        # Find the maximum score in the DP table
        max_score = np.max(dp)
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        # Traceback from the maximum score position
        aligned_a, aligned_b = [], []
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i, j] == dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif i > 0 and dp[i, j] == dp[i-1, j] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            else:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
        # Reverse to get the correct order
        aligned_a = aligned_a[::-1]
        aligned_b = aligned_b[::-1]
        # Return the score and aligned sequences
        return (max_score, "".join(aligned_a), "".join(aligned_b))
    
    # Handle global mode
    else:
        # Traceback from the end of the sequences
        i, j = len_a, len_b
        aligned_a, aligned_b = [], []
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i, j] == dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif i > 0 and dp[i, j] == dp[i-1, j] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            else:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
        # Reverse to get the correct order
        aligned_a = aligned_a[::-1]
        aligned_b = aligned_b[::-1]
        # Return the score and aligned sequences
        return (dp[len_a, len_b], "".join(aligned_a), "".join(aligned_b))