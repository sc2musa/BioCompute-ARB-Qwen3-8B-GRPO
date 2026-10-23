import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)

    # Initialize the DP table based on mode
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
            # Score for match/mismatch
            score = dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            
            # Score for gap in seq_a
            score_gap_a = dp[i-1, j] + gap_open if i == 1 else dp[i-1, j] + gap_extend
            
            # Score for gap in seq_b
            score_gap_b = dp[i, j-1] + gap_open if j == 1 else dp[i, j-1] + gap_extend
            
            # Choose the maximum score
            dp[i, j] = max(score, score_gap_a, score_gap_b)

    # Backtrack to find the optimal alignment
    i, j = len_a, len_b
    aligned_a, aligned_b = [], []
    
    while i > 0 and j > 0:
        if dp[i, j] == dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif dp[i, j] == dp[i-1, j] + gap_extend if i > 1 else dp[i-1, j] + gap_open:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        else:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1

    # Reverse the alignment to get the correct order
    aligned_a = ''.join(reversed(aligned_a))
    aligned_b = ''.join(reversed(aligned_b))

    # Handle local mode
    if mode == "local":
        # Find the maximum score in the DP table
        max_score = np.max(dp)
        
        # If no positive-scoring local alignment, return (0, "", "")
        if max_score <= 0:
            return (0, "", "")
        
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        
        # Backtrack from the maximum score to get the alignment
        aligned_a, aligned_b = [], []
        while i > 0 and j > 0:
            if dp[i, j] == dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif dp[i, j] == dp[i-1, j] + gap_extend if i > 1 else dp[i-1, j] + gap_open:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            else:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
        
        # Reverse the alignment to get the correct order
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
        
        # Ensure the alignment starts and ends with a match
        # (This is a simplified approach for local alignment)
        # For a complete implementation, additional checks would be needed
        
        # Return the score and the aligned sequences
        return (max_score, aligned_a, aligned_b)
    
    # Return the score and the aligned sequences
    return (dp[len_a, len_b], aligned_a, aligned_b)