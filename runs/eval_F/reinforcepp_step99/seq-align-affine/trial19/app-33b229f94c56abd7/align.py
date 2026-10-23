import numpy as np

def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Handle empty sequences
    if not seq_a and not seq_b:
        return (0, "", "")
    if not seq_a:
        return (0, "", seq_b)
    if not seq_b:
        return (0, seq_a, "")
    
    # Initialize DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize first row and column
    for i in range(1, len_a + 1):
        dp[i, 0] = dp[i-1, 0] + gap_open + (i-1)*gap_extend
    for j in range(1, len_b + 1):
        dp[0, j] = dp[0, j-1] + gap_open + (j-1)*gap_extend
    
    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Match/mismatch
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            dp[i, j] = max(
                dp[i-1, j-1] + match_score,
                # Gap in seq_a
                dp[i-1, j] + gap_extend,
                # Gap in seq_b
                dp[i, j-1] + gap_extend
            )
    
    # Find optimal score
    if mode == "global":
        score = dp[len_a, len_b]
    else:
        # Find maximum score in the table
        score = np.max(dp)
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        # Truncate sequences to the matching region
        seq_a = seq_a[:i]
        seq_b = seq_b[:j]
        # Recalculate the score for the truncated sequences
        score = dp[i, j]
    
    # Traceback to find the alignment
    aligned_a, aligned_b = "", ""
    i, j = len_a, len_b
    while i > 0 and j > 0:
        if seq_a[i-1] == seq_b[j-1]:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        elif dp[i-1, j] == dp[i, j]:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
        else:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
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
    
    return (score, aligned_a, aligned_b)