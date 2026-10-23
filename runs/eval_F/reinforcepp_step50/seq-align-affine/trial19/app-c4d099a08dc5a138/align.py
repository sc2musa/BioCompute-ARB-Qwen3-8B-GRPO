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
    for i in range(1, len_a + 1):
        dp[i, 0] = gap_open + (i - 1) * gap_extend
    for j in range(1, len_b + 1):
        dp[0, j] = gap_open + (j - 1) * gap_extend
    
    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate score for match/mismatch
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Calculate gap penalties
            gap_a = gap_open + (i - 1) * gap_extend
            gap_b = gap_open + (j - 1) * gap_extend
            
            # Choose maximum value
            dp[i, j] = max(
                dp[i-1, j-1] + score,
                dp[i-1, j] + gap_extend,
                dp[i, j-1] + gap_extend
            )
    
    # Handle local mode
    if mode == "local":
        # Find the maximum score in the DP table
        max_score = np.max(dp)
        
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        
        # Initialize the DP table for local mode
        local_dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
        
        # Fill the local DP table
        for i2 in range(i, len_a + 1):
            for j2 in range(j, len_b + 1):
                # Calculate score for match/mismatch
                score = match if seq_a[i2-1] == seq_b[j2-1] else mismatch
                
                # Calculate gap penalties
                gap_a = gap_open + (i2 - 1) * gap_extend
                gap_b = gap_open + (j2 - 1) * gap_extend
                
                # Choose maximum value
                local_dp[i2, j2] = max(
                    local_dp[i2-1, j2-1] + score,
                    local_dp[i2-1, j2] + gap_extend,
                    local_dp[i2, j2-1] + gap_extend
                )
        
        # Traceback to find alignment
        i, j = len_a, len_b
        aligned_a, aligned_b = "", ""
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and local_dp[i, j] == local_dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif i > 0 and local_dp[i, j] == local_dp[i-1, j] + gap_extend:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            elif j > 0 and local_dp[i, j] == local_dp[i, j-1] + gap_extend:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
        
        return (max_score, aligned_a, aligned_b)
    
    # Handle global mode
    else:
        # Traceback to find alignment
        i, j = len_a, len_b
        aligned_a, aligned_b = "", ""
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i, j] == dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
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