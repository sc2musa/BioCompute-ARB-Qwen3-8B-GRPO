import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize the DP table based on the mode
    if mode == "global":
        # Needleman-Wunsch initialization
        dp[0, :] = [i * gap_open for i in range(len_b + 1)]
        dp[:, 0] = [i * gap_open for i in range(len_a + 1)]
    elif mode == "local":
        # Smith-Waterman initialization
        dp[0, :] = [0] * (len_b + 1)
        dp[:, 0] = [0] * (len_a + 1)
    else:
        raise ValueError("Invalid mode. Use 'global' or 'local'.")
    
    # Fill the DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Score for matching or mismatching characters
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Score for aligning with a gap in seq_a
            score_gap_a = dp[i-1, j] + gap_extend
            
            # Score for aligning with a gap in seq_b
            score_gap_b = dp[i, j-1] + gap_extend
            
            # Score for aligning both characters
            score_match = dp[i-1, j-1] + score
            
            # Choose the maximum score
            dp[i, j] = max(score_gap_a, score_gap_b, score_match)
    
    # Determine the optimal score based on the mode
    if mode == "global":
        score = dp[len_a, len_b]
    elif mode == "local":
        # Find the maximum score in the DP table
        score = np.max(dp)
        # If no positive-scoring local alignment exists, return (0, "", "")
        if score <= 0:
            return (0, "", "")
    else:
        raise ValueError("Invalid mode. Use 'global' or 'local'.")
    
    # Traceback to find the optimal alignment
    aligned_a = []
    aligned_b = []
    i, j = len_a, len_b
    
    while i > 0 and j > 0:
        if mode == "global":
            # Global alignment: trace back to the start
            if dp[i, j] == dp[i-1, j] + gap_extend:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
            elif dp[i, j] == dp[i, j-1] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            else:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
        elif mode == "local":
            # Local alignment: trace back to the start of the maximum score
            if dp[i, j] == dp[i-1, j] + gap_extend:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
            elif dp[i, j] == dp[i, j-1] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            else:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
    
    # Reverse the aligned sequences to get the correct order
    aligned_a = ''.join(reversed(aligned_a))
    aligned_b = ''.join(reversed(aligned_b))
    
    return (score, aligned_a, aligned_b)