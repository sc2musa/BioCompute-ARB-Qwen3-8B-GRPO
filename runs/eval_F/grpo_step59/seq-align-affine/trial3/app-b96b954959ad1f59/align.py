import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the DP table
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m+1, n+1), dtype=int)
    
    # Initialize the first row and column
    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] + gap_open + (i-1)*gap_extend
    for j in range(1, n+1):
        dp[0][j] = dp[0][j-1] + gap_open + (j-1)*gap_extend
    
    # Handle local mode
    if mode == "local":
        # Initialize the DP table with -infinity
        dp = np.full((m+1, n+1), -np.inf, dtype=int)
        # Set the initial cell to 0
        dp[0][0] = 0
        
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
            
            # Calculate the score for gaps
            gap_score = gap_open + (i-1)*gap_extend
            
            # Choose the best score
            dp[i][j] = max(
                dp[i-1][j-1] + score,
                dp[i-1][j] + gap_score,
                dp[i][j-1] + gap_score
            )
    
    # Extract the optimal alignment
    i, j = m, n
    aligned_a = []
    aligned_b = []
    
    # Trace back through the DP table
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + gap_open + (i-1)*gap_extend:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + gap_open + (j-1)*gap_extend:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
    
    # Reverse the alignment to get the correct order
    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]
    
    # Return the score and alignments
    return (dp[m][n], "".join(aligned_a), "".join(aligned_b))