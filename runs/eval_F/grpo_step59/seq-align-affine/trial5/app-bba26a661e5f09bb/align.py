import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Handle empty sequences
    if not seq_a or not seq_b:
        return (0, seq_a, seq_b)
    
    # Initialize matrix
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m+1, n+1))
    
    # Fill first row and column
    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] + gap_open + (i-1)*gap_extend
    for j in range(1, n+1):
        dp[0][j] = dp[0][j-1] + gap_open + (j-1)*gap_extend
    
    # Fill matrix
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Calculate score for match/mismatch
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Calculate score for gap in seq_a
            gap_a = dp[i-1][j] + gap_open + (i-1)*gap_extend
            
            # Calculate score for gap in seq_b
            gap_b = dp[i][j-1] + gap_open + (j-1)*gap_extend
            
            # Choose the best score
            dp[i][j] = max(score, gap_a, gap_b)
    
    # Trace back to find the alignment
    i, j = m, n
    aligned_a = []
    aligned_b = []
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1] and dp[i][j] == dp[i-1][j-1] + match:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i-1][j] + gap_open + (i-1)*gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            elif dp[i][j] == dp[i][j-1] + gap_open + (j-1)*gap_extend:
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
    
    # Reverse the alignment
    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]
    
    # Return the result
    return (dp[m][n], "".join(aligned_a), "".join(aligned_b))