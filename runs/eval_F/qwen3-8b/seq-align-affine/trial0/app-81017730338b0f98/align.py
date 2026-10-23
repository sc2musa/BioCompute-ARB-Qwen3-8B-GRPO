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
    
    # Initialize DP tables
    m, n = len(seq_a), len(seq_b)
    
    # Create DP tables for global and local alignment
    if mode == "global":
        # Needleman-Wunsch algorithm
        # Initialize DP table
        dp = np.zeros((m+1, n+1), dtype=int)
        
        # Initialize first row and column
        for i in range(1, m+1):
            dp[i][0] = dp[i-1][0] + gap_open + (i-1)*gap_extend
        for j in range(1, n+1):
            dp[0][j] = dp[0][j-1] + gap_open + (j-1)*gap_extend
        
        # Fill DP table
        for i in range(1, m+1):
            for j in range(1, n+1):
                # Match/mismatch
                score = dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
                
                # Gap in seq_a
                score_gap_a = dp[i-1][j] + gap_extend
                
                # Gap in seq_b
                score_gap_b = dp[i][j-1] + gap_extend
                
                # Choose maximum score
                dp[i][j] = max(score, score_gap_a, score_gap_b)
        
        # Backtrack to find the alignment
        i, j = m, n
        aligned_a, aligned_b = "", ""
        while i > 0 and j > 0:
            if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i-1][j] + gap_extend:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            else:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
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
        
        return (dp[m][n], aligned_a, aligned_b)
    elif mode == "local":
        # Smith-Waterman algorithm
        # Initialize DP table
        dp = np.zeros((m+1, n+1), dtype=int)
        
        # Fill DP table
        for i in range(1, m+1):
            for j in range(1, n+1):
                # Match/mismatch
                score = dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
                
                # Gap in seq_a
                score_gap_a = dp[i-1][j] + gap_extend
                
                # Gap in seq_b
                score_gap_b = dp[i][j-1] + gap_extend
                
                # Choose maximum score (but can't be negative)
                dp[i][j] = max(score, score_gap_a, score_gap_b, 0)
        
        # Find the maximum score and its position
        max_score = dp.max()
        if max_score == 0:
            return (0, "", "")
        
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        
        # Backtrack to find the alignment
        aligned_a, aligned_b = "", ""
        while i > 0 and j > 0 and dp[i][j] > 0:
            if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i-1][j] + gap_extend:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            else:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
        
        return (max_score, aligned_a, aligned_b)
    else:
        raise ValueError("Invalid mode. Use 'global' or 'local'.")