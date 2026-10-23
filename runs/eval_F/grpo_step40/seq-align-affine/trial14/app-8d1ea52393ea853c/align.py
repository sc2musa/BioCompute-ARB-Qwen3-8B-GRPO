import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Handle empty sequences
    if not seq_a:
        return (0, seq_b, "")
    if not seq_b:
        return (0, "", seq_a)
    
    # Initialize DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m+1, n+1), dtype=int)
    
    # Initialize first row and column
    for i in range(1, m+1):
        dp[i][0] = i * gap_open + (i-1) * gap_extend
    for j in range(1, n+1):
        dp[0][j] = j * gap_open + (j-1) * gap_extend
    
    # Fill DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Calculate score for match/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Calculate score for gap in seq_a
            gap_a = dp[i-1][j] + gap_extend
            # Calculate score for gap in seq_b
            gap_b = dp[i][j-1] + gap_extend
            
            # Choose the best option
            dp[i][j] = max(score + dp[i-1][j-1], gap_a, gap_b)
    
    # Handle local mode
    if mode == "local":
        # Find the maximum score in the DP table
        max_score = np.max(dp)
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        # Traceback from the maximum score
        aligned_a, aligned_b = "", ""
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                if seq_a[i-1] == seq_b[j-1]:
                    aligned_a = seq_a[i-1] + aligned_a
                    aligned_b = seq_b[j-1] + aligned_b
                    i -= 1
                    j -= 1
                else:
                    # Check if we should take a diagonal move
                    if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                        aligned_a = seq_a[i-1] + aligned_a
                        aligned_b = seq_b[j-1] + aligned_b
                        i -= 1
                        j -= 1
                    else:
                        # Check if we should take a gap in seq_a
                        if dp[i][j] == dp[i-1][j] + gap_extend:
                            aligned_a = "-" + aligned_a
                            aligned_b = seq_b[j-1] + aligned_b
                            i -= 1
                        else:
                            # Check if we should take a gap in seq_b
                            aligned_a = seq_a[i-1] + aligned_a
                            aligned_b = "-" + aligned_b
                            j -= 1
            elif i > 0:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
            else:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                j -= 1
        return (max_score, aligned_a, aligned_b)
    else:
        # Handle global mode
        i, j = m, n
        aligned_a, aligned_b = "", ""
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                if seq_a[i-1] == seq_b[j-1]:
                    aligned_a = seq_a[i-1] + aligned_a
                    aligned_b = seq_b[j-1] + aligned_b
                    i -= 1
                    j -= 1
                else:
                    # Check if we should take a diagonal move
                    if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                        aligned_a = seq_a[i-1] + aligned_a
                        aligned_b = seq_b[j-1] + aligned_b
                        i -= 1
                        j -= 1
                    else:
                        # Check if we should take a gap in seq_a
                        if dp[i][j] == dp[i-1][j] + gap_extend:
                            aligned_a = "-" + aligned_a
                            aligned_b = seq_b[j-1] + aligned_b
                            i -= 1
                        else:
                            # Check if we should take a gap in seq_b
                            aligned_a = seq_a[i-1] + aligned_a
                            aligned_b = "-" + aligned_b
                            j -= 1
            elif i > 0:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
            else:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                j -= 1
        return (dp[m][n], aligned_a, aligned_b)