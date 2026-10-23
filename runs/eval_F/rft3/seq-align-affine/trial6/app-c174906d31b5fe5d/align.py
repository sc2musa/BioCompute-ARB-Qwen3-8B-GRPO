import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize the first row and column
    for i in range(len_a + 1):
        dp[i, 0] = i * gap_open + (i - 1) * gap_extend
    for j in range(len_b + 1):
        dp[0, j] = j * gap_open + (j - 1) * gap_extend
    
    # Fill the DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate the score for matching characters
            if seq_a[i-1] == seq_b[j-1]:
                match_score = match
            else:
                match_score = mismatch
            
            # Calculate the score for the current cell
            dp[i, j] = max(
                dp[i-1, j-1] + match_score,  # Match
                dp[i-1, j] + gap_extend,  # Gap in seq_b
                dp[i, j-1] + gap_extend  # Gap in seq_a
            )
    
    # Find the optimal alignment
    if mode == "global":
        # The score is the value in the bottom-right cell
        score = dp[len_a, len_b]
        
        # Backtrack to find the aligned sequences
        aligned_a = ""
        aligned_b = ""
        i, j = len_a, len_b
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif dp[i-1, j] == dp[i, j]:
                # Gap in seq_b
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
            elif dp[i, j-1] == dp[i, j]:
                # Gap in seq_a
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            else:
                # This should not happen with proper implementation
                raise ValueError("Invalid DP table state")
        
        # Handle remaining gaps
        while i > 0:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        while j > 0:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
    else:
        # Local mode: find the maximum score in the table
        max_score = np.max(dp)
        score = max_score
        
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        
        # Backtrack to find the aligned sequences
        aligned_a = ""
        aligned_b = ""
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif dp[i-1, j] == dp[i, j]:
                # Gap in seq_b
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
            elif dp[i, j-1] == dp[i, j]:
                # Gap in seq_a
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            else:
                # This should not happen with proper implementation
                raise ValueError("Invalid DP table state")
        
        # Handle remaining gaps
        while i > 0:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        while j > 0:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
    
    return (score, aligned_a, aligned_b)