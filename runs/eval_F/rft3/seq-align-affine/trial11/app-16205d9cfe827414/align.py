import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1))
    
    # Initialize first row and column
    for i in range(len_a + 1):
        dp[i, 0] = gap_open + (i - 1) * gap_extend
    for j in range(len_b + 1):
        dp[0, j] = gap_open + (j - 1) * gap_extend
    
    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate scores for match/mismatch and gaps
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            gap_a = dp[i-1, j] + gap_extend
            gap_b = dp[i, j-1] + gap_extend
            
            # Choose the best score
            dp[i, j] = max(match_score, gap_a, gap_b)
    
    # Traceback to find the optimal alignment
    i, j = len_a, len_b
    aligned_a = []
    aligned_b = []
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            else:
                # Choose the best path
                if dp[i, j] == dp[i-1, j] + gap_extend:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    i -= 1
                else:
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
    
    # Return the score and aligned sequences
    return dp[len_a, len_b], "".join(aligned_a), "".join(aligned_b)