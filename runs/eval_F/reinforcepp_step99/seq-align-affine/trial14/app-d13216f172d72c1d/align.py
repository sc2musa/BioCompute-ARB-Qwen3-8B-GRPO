import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Handle empty sequences
    if not seq_a:
        return (0, seq_b, "")
    if not seq_b:
        return (0, "", seq_a)
    
    # Initialize the DP table
    len_a = len(seq_a)
    len_b = len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Fill the DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate the score for matching/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Calculate the best score from the three possible directions
            dp[i][j] = max(
                dp[i-1][j] + gap_extend,  # Gap in seq_a
                dp[i][j-1] + gap_extend,  # Gap in seq_b
                dp[i-1][j-1] + score      # Match/mismatch
            )
            
            # Handle the first gap in a sequence
            if i == 1 and j == 1:
                dp[i][j] = max(dp[i][j], gap_open)
            elif i == 1:
                dp[i][j] = max(dp[i][j], dp[i][j-1] + gap_extend)
            elif j == 1:
                dp[i][j] = max(dp[i][j], dp[i-1][j] + gap_extend)
    
    # Find the optimal alignment
    if mode == "global":
        # Find the maximum score in the last row/column
        max_score = dp[len_a, len_b]
        # Backtrack to find the alignment
        i, j = len_a, len_b
        aligned_a = ""
        aligned_b = ""
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif dp[i-1][j] + gap_extend == dp[i][j]:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
            elif dp[i][j-1] + gap_extend == dp[i][j]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                j -= 1
            else:
                # This should not happen with the correct implementation
                aligned_a = "-" + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
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
        return (max_score, aligned_a, aligned_b)
    else:
        # Find the maximum score in the entire table
        max_score = np.max(dp)
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        # Backtrack to find the alignment
        aligned_a = ""
        aligned_b = ""
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif dp[i-1][j] + gap_extend == dp[i][j]:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
            elif dp[i][j-1] + gap_extend == dp[i][j]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                j -= 1
            else:
                # This should not happen with the correct implementation
                aligned_a = "-" + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
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
        return (max_score, aligned_a, aligned_b)