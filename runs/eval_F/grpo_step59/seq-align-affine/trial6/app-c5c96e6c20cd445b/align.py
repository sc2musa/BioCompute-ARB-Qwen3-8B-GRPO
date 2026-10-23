import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m+1, n+1))
    
    # Initialize the first row and column
    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] + gap_open + (i-1)*gap_extend
    for j in range(1, n+1):
        dp[0][j] = dp[0][j-1] + gap_open + (j-1)*gap_extend
    
    # Fill the DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Calculate the score for matching/mismatch
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Calculate the score for gap in seq_a
            score_gap_a = dp[i-1][j] + gap_extend
            
            # Calculate the score for gap in seq_b
            score_gap_b = dp[i][j-1] + gap_extend
            
            # Choose the maximum score
            dp[i][j] = max(score, score_gap_a, score_gap_b)
    
    # Backtrack to find the optimal alignment
    i, j = m, n
    aligned_a = []
    aligned_b = []
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            # Check if the current score is from a match/mismatch
            if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            else:
                # Check if the current score is from a gap in seq_a
                if dp[i][j] == dp[i-1][j] + gap_extend:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    i -= 1
                else:
                    # Check if the current score is from a gap in seq_b
                    aligned_a.append("-")
                    aligned_b.append(seq_b[j-1])
                    j -= 1
        elif i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        else:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
    
    # Reverse the alignments
    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]
    
    # Return the score and alignments
    return (dp[m][n], "".join(aligned_a), "".join(aligned_b))