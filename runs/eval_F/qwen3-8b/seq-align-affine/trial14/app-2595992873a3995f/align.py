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
    
    # Initialize DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    # Initialize first row and column
    for i in range(m + 1):
        if mode == "global":
            dp[i][0] = i * gap_open + (i - 1) * gap_extend if i > 0 else 0
        else:
            dp[i][0] = 0
    for j in range(n + 1):
        if mode == "global":
            dp[0][j] = j * gap_open + (j - 1) * gap_extend if j > 0 else 0
        else:
            dp[0][j] = 0
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Match/mismatch score
            match_score = dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            
            # Gap in seq_a
            gap_a_score = dp[i-1][j] + gap_open + gap_extend
            
            # Gap in seq_b
            gap_b_score = dp[i][j-1] + gap_open + gap_extend
            
            # Choose the maximum score
            if mode == "global":
                dp[i][j] = max(match_score, gap_a_score, gap_b_score)
            else:
                # For local alignment, use max of 0 and the scores
                dp[i][j] = max(0, match_score, gap_a_score, gap_b_score)
    
    # Find the optimal score and position for local alignment
    if mode == "local":
        max_score = dp[m][n]
        # Find the position with the maximum score
        for i in range(m + 1):
            for j in range(n + 1):
                if dp[i][j] == max_score:
                    # Start backtracking from this position
                    i_max, j_max = i, j
                    break
            else:
                continue
            break
        # Initialize the DP table for backtracking
        dp = np.zeros((m + 1, n + 1), dtype=int)
        # Initialize first row and column
        for i in range(m + 1):
            dp[i][0] = 0
        for j in range(n + 1):
            dp[0][j] = 0
        # Fill DP table again for backtracking
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Match/mismatch score
                match_score = dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
                
                # Gap in seq_a
                gap_a_score = dp[i-1][j] + gap_open + gap_extend
                
                # Gap in seq_b
                gap_b_score = dp[i][j-1] + gap_open + gap_extend
                
                # Choose the maximum score
                dp[i][j] = max(0, match_score, gap_a_score, gap_b_score)
        # Backtrack from the position with the maximum score
        i, j = i_max, j_max
        aligned_a, aligned_b = ",", ","
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + gap_open + gap_extend:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            elif j > 0 and dp[i][j] == dp[i][j-1] + gap_open + gap_extend:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
            else:
                break
        
        # Return the score and aligned sequences
        return (max_score, aligned_a, aligned_b)
    
    # Backtrack to find the optimal alignment for global mode
    i, j = m, n
    aligned_a, aligned_b = ",", ","
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + gap_open + gap_extend:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + gap_open + gap_extend:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
        else:
            # If we reach here, it means we have a zero score and need to break
            if dp[i][j] == 0:
                break
            # Otherwise, we might have an error, so break
            break
    
    # Return the score and aligned sequences
    return (dp[m][n], aligned_a, aligned_b)