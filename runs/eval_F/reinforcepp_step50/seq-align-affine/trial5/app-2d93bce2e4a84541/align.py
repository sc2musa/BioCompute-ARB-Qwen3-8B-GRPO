import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Handle empty sequences
    if not seq_a or not seq_b:
        return (0, "", "")
    
    # Initialize DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize first row and column with gap penalties
    for i in range(1, len_a + 1):
        dp[i][0] = gap_open + (i - 1) * gap_extend
    for j in range(1, len_b + 1):
        dp[0][j] = gap_open + (j - 1) * gap_extend
    
    # Fill the DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate score for match/mismatch
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Calculate score for gap in seq_a
            score_gap_a = gap_open + (i - 1) * gap_extend
            
            # Calculate score for gap in seq_b
            score_gap_b = gap_open + (j - 1) * gap_extend
            
            # Take the maximum of the three options
            dp[i][j] = max(dp[i-1][j-1] + score, dp[i-1][j] + score_gap_a, dp[i][j-1] + score_gap_b)
    
    # Find the optimal score
    if mode == "global":
        score = dp[len_a][len_b]
    else:
        # For local mode, find the maximum score in the table
        score = np.max(dp)
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        
        # Extract the substring sequences
        seq_a = seq_a[i-100:i+100] if i-100 >= 0 else seq_a[0:i+100]
        seq_b = seq_b[j-100:j+100] if j-100 >= 0 else seq_b[0:j+100]
        
        # Re-initialize the DP table for the substring
        len_a, len_b = len(seq_a), len(seq_b)
        dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
        
        # Initialize first row and column with gap penalties
        for i in range(1, len_a + 1):
            dp[i][0] = gap_open + (i - 1) * gap_extend
        for j in range(1, len_b + 1):
            dp[0][j] = gap_open + (j - 1) * gap_extend
        
        # Fill the DP table
        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                # Calculate score for match/mismatch
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                
                # Calculate score for gap in seq_a
                score_gap_a = gap_open + (i - 1) * gap_extend
                
                # Calculate score for gap in seq_b
                score_gap_b = gap_open + (j - 1) * gap_extend
                
                # Take the maximum of the three options
                dp[i][j] = max(dp[i-1][j-1] + score, dp[i-1][j] + score_gap_a, dp[i][j-1] + score_gap_b)
        
        # Get the final score
        score = dp[len_a][len_b]
    
    # Trace back to find the alignment
    aligned_a = ""
    aligned_b = ""
    i, j = len_a, len_b
    while i > 0 and j > 0:
        # Check if the current cell was filled by a match/mismatch
        if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        # Check if the current cell was filled by a gap in seq_a
        elif dp[i][j] == dp[i-1][j] + gap_open + (i - 1) * gap_extend:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
        # Check if the current cell was filled by a gap in seq_b
        elif dp[i][j] == dp[i][j-1] + gap_open + (j - 1) * gap_extend:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            j -= 1
    
    # Handle remaining characters
    while i > 0:
        aligned_a = "-" + aligned_a
        aligned_b = seq_b[j-1] + aligned_b
        i -= 1
    while j > 0:
        aligned_a = seq_a[i-1] + aligned_a
        aligned_b = "-" + aligned_b
        j -= 1
    
    return (score, aligned_a, aligned_b)