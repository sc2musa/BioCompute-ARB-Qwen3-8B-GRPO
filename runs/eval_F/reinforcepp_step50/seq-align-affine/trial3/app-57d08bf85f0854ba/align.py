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
        dp[i, 0] = dp[i-1, 0] + gap_open + (i-1)*gap_extend
    for j in range(1, len_b + 1):
        dp[0, j] = dp[0, j-1] + gap_open + (j-1)*gap_extend
    
    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate score for match/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Calculate score for gap in seq_a
            gap_a = dp[i-1, j] + gap_open + (i-1)*gap_extend
            
            # Calculate score for gap in seq_b
            gap_b = dp[i, j-1] + gap_open + (j-1)*gap_extend
            
            # Choose the maximum score
            dp[i, j] = max(score + dp[i-1, j-1], gap_a, gap_b)
    
    # Handle local mode
    if mode == "local":
        # Find the maximum score in the DP table
        max_score = np.max(dp)
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        # Traceback from the maximum score
        aligned_a, aligned_b = "", ""
        
        while i > 0 and j > 0:
            if dp[i, j] == dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                # Match/mismatch
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif dp[i, j] == dp[i-1, j] + gap_open + (i-1)*gap_extend:
                # Gap in seq_a
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
            else:
                # Gap in seq_b
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                j -= 1
        
        # Handle remaining characters
        while i > 0:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        while j > 0:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
        
        return max_score, aligned_a, aligned_b
    
    # Handle global mode
    else:
        # Traceback from the end
        i, j = len_a, len_b
        aligned_a, aligned_b = "", ""
        
        while i > 0 and j > 0:
            if dp[i, j] == dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                # Match/mismatch
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif dp[i, j] == dp[i-1, j] + gap_open + (i-1)*gap_extend:
                # Gap in seq_a
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
            else:
                # Gap in seq_b
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                j -= 1
        
        # Handle remaining characters
        while i > 0:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        while j > 0:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
        
        return dp[len_a, len_b], aligned_a, aligned_b