import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Handle empty sequences
    if not seq_a or not seq_b:
        return (0, seq_a, seq_b)
    
    # Initialize DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize first row and column
    for i in range(1, len_a + 1):
        dp[i][0] = gap_open + (i - 1) * gap_extend
    for j in range(1, len_b + 1):
        dp[0][j] = gap_open + (j - 1) * gap_extend
    
    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate score for match/mismatch
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Calculate score for gap in seq_a
            gap_a = dp[i-1][j] + gap_open + (i - 1) * gap_extend
            
            # Calculate score for gap in seq_b
            gap_b = dp[i][j-1] + gap_open + (j - 1) * gap_extend
            
            # Choose maximum score
            dp[i][j] = max(
                dp[i-1][j-1] + score,
                gap_a,
                gap_b
            )
    
    # Handle local mode
    if mode == "local":
        # Find maximum score in DP table
        max_score = np.max(dp)
        
        # Find position of maximum score
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if dp[i][j] == max_score:
                    # Backtrack to find alignment
                    aligned_a = ""
                    aligned_b = ""
                    while i > 0 and j > 0:
                        if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                            aligned_a += seq_a[i-1]
                            aligned_b += seq_b[j-1]
                            i -= 1
                            j -= 1
                        elif dp[i][j] == dp[i-1][j] + gap_open + (i - 1) * gap_extend:
                            aligned_a += seq_a[i-1]
                            aligned_b += "-"
                            i -= 1
                        else:
                            aligned_a += "-"
                            aligned_b += seq_b[j-1]
                            j -= 1
                    
                    # Return result
                    return (max_score, aligned_a[::-1], aligned_b[::-1])
        
        # If no positive score found
        return (0, "", "")
    
    # Handle global mode
    else:
        # Backtrack to find alignment
        aligned_a = ""
        aligned_b = ""
        i, j = len_a, len_b
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                    aligned_a += seq_a[i-1]
                    aligned_b += seq_b[j-1]
                    i -= 1
                    j -= 1
                elif dp[i][j] == dp[i-1][j] + gap_open + (i - 1) * gap_extend:
                    aligned_a += seq_a[i-1]
                    aligned_b += "-"
                    i -= 1
                else:
                    aligned_a += "-"
                    aligned_b += seq_b[j-1]
                    j -= 1
            elif i > 0:
                aligned_a += seq_a[i-1]
                aligned_b += "-"
                i -= 1
            else:
                aligned_a += "-"
                aligned_b += seq_b[j-1]
                j -= 1
        
        # Return result
        return (dp[len_a][len_b], aligned_a[::-1], aligned_b[::-1])