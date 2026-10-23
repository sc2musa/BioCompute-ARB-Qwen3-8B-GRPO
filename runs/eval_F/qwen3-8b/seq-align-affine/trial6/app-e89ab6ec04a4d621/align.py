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
    
    # Initialize the DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize the first row and column based on the mode
    if mode == "global":
        # Needleman-Wunsch initialization
        for i in range(len_a + 1):
            dp[i][0] = i * gap_open + (i - 1) * gap_extend
        for j in range(len_b + 1):
            dp[0][j] = j * gap_open + (j - 1) * gap_extend
    elif mode == "local":
        # Smith-Waterman initialization
        for i in range(len_a + 1):
            dp[i][0] = 0
        for j in range(len_b + 1):
            dp[0][j] = 0
    else:
        raise ValueError("Invalid mode. Use 'global' or 'local'.")
    
    # Fill the DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Score for matching/mismatch
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Score for aligning seq_a[i-1] with a gap in seq_b
            score_gap_b = dp[i-1][j] + gap_extend
            
            # Score for aligning seq_b[j-1] with a gap in seq_a
            score_gap_a = dp[i][j-1] + gap_extend
            
            # Score for aligning both sequences
            score_match = dp[i-1][j-1] + match_score
            
            # Choose the maximum score
            if mode == "global":
                dp[i][j] = max(score_match, score_gap_a, score_gap_b)
            elif mode == "local":
                dp[i][j] = max(score_match, score_gap_a, score_gap_b, 0)
    
    # Determine the optimal score
    if mode == "global":
        score = dp[len_a][len_b]
    elif mode == "local":
        score = np.max(dp)
    else:
        raise ValueError("Invalid mode. Use 'global' or 'local'.")
    
    # Trace back to find the optimal alignment
    aligned_a = []
    aligned_b = []
    i, j = len_a, len_b
    
    while i > 0 and j > 0:
        if mode == "global":
            # Check for match/mismatch
            if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i][j-1] + gap_extend:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
            else:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
        elif mode == "local":
            # Check for match/mismatch
            if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i][j-1] + gap_extend:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
            elif dp[i][j] == dp[i-1][j] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            else:
                # If the score is 0, we've reached the end of the local alignment
                break
    
    # Reverse the aligned sequences
    aligned_a = ''.join(reversed(aligned_a))
    aligned_b = ''.join(reversed(aligned_b))
    
    return (score, aligned_a, aligned_b)