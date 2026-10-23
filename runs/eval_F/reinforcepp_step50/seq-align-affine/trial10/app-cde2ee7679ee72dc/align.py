import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Handle empty sequences
    if not seq_a or not seq_b:
        return (0, "", "")
    
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
            # Calculate scores for different possibilities
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Option 1: match/mismatch
            option1 = dp[i-1][j-1] + match_score
            
            # Option 2: gap in seq_a
            option2 = dp[i-1][j] + gap_extend
            
            # Option 3: gap in seq_b
            option3 = dp[i][j-1] + gap_extend
            
            # Choose the best option
            dp[i][j] = max(option1, option2, option3)
    
    # Find the maximum score
    max_score = dp[len_a][len_b]
    
    # Backtrack to find the alignment
    aligned_a = ""
    aligned_b = ""
    i, j = len_a, len_b
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + gap_extend:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + gap_extend:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
        else:
            # Handle edge cases
            if i > 0:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            elif j > 0:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
            else:
                break
    
    return (max_score, aligned_a, aligned_b)