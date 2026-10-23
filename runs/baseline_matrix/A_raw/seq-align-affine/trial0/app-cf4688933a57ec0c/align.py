import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize the first row and column based on mode
    if mode == "global":
        # Needleman-Wunsch initialization
        dp[0, :] = [i * gap_open for i in range(len_b + 1)]
        dp[:, 0] = [i * gap_open for i in range(len_a + 1)]
    else:
        # Smith-Waterman initialization
        dp[0, :] = [0] * (len_b + 1)
        dp[:, 0] = [0] * (len_a + 1)
    
    # Fill the DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Score for matching/mismatch
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Score for alignment from the top (gap in seq_b)
            score_top = dp[i-1, j] + gap_extend
            
            # Score for alignment from the left (gap in seq_a)
            score_left = dp[i, j-1] + gap_extend
            
            # Score for alignment from the diagonal (match/mismatch)
            score_diag = dp[i-1, j-1] + score
            
            # Choose the maximum score
            dp[i, j] = max(score_top, score_left, score_diag)
            
            # If in local mode, set to 0 if negative
            if mode == "local" and dp[i, j] < 0:
                dp[i, j] = 0
    
    # Determine the score based on mode
    if mode == "global":
        score = dp[len_a, len_b]
    else:
        # Find the maximum score in the DP table for local mode
        score = np.max(dp)
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
    
    # Backtrack to find the alignment
    aligned_a = []
    aligned_b = []
    
    if mode == "global":
        i, j = len_a, len_b
        while i > 0 and j > 0:
            # Check if the current cell is from a diagonal move (match/mismatch)
            if dp[i, j] == dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            # Check if the current cell is from a top move (gap in seq_b)
            elif dp[i, j] == dp[i-1, j] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            # Check if the current cell is from a left move (gap in seq_a)
            else:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
        # Reverse the aligned sequences to get the correct order
        aligned_a = aligned_a[::-1]
        aligned_b = aligned_b[::-1]
    else:
        # Backtrack from the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        while i > 0 and j > 0:
            # Check if the current cell is from a diagonal move (match/mismatch)
            if dp[i, j] == dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            # Check if the current cell is from a top move (gap in seq_b)
            elif dp[i, j] == dp[i-1, j] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            # Check if the current cell is from a left move (gap in seq_a)
            else:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
        # Reverse the aligned sequences to get the correct order
        aligned_a = aligned_a[::-1]
        aligned_b = aligned_b[::-1]
    
    # Return the score and aligned sequences
    return (score, ''.join(aligned_a), ''.join(aligned_b))