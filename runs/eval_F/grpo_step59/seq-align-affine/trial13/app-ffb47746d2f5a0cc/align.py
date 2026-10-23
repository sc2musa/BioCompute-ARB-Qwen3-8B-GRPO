import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize DP table
    len_a = len(seq_a)
    len_b = len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize first row and column with gap penalties
    for i in range(1, len_a + 1):
        dp[i][0] = dp[i-1][0] + gap_extend
    for j in range(1, len_b + 1):
        dp[0][j] = dp[0][j-1] + gap_extend
    
    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate score for match/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Calculate score for gap in seq_a
            if i > 1:
                gap_a = dp[i-1][j] + gap_extend
            else:
                gap_a = gap_open
            
            # Calculate score for gap in seq_b
            if j > 1:
                gap_b = dp[i][j-1] + gap_extend
            else:
                gap_b = gap_open
            
            # Choose the maximum score
            dp[i][j] = max(score + dp[i-1][j-1], gap_a, gap_b)
    
    # Handle local mode
    if mode == "local":
        max_score = np.max(dp)
        # Find the position with max score
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if dp[i][j] == max_score:
                    # Trace back from this position
                    aligned_a = []
                    aligned_b = []
                    i_tmp, j_tmp = i, j
                    while i_tmp > 0 or j_tmp > 0:
                        if i_tmp > 0 and j_tmp > 0:
                            # Check if this cell is a match/mismatch
                            if dp[i_tmp][j_tmp] == dp[i_tmp-1][j_tmp-1] + (match if seq_a[i_tmp-1] == seq_b[j_tmp-1] else mismatch):
                                aligned_a.append(seq_a[i_tmp-1])
                                aligned_b.append(seq_b[j_tmp-1])
                                i_tmp -= 1
                                j_tmp -= 1
                            else:
                                # Check if this cell is a gap in seq_a
                                if i_tmp > 1 and dp[i_tmp][j_tmp] == dp[i_tmp-1][j_tmp] + gap_extend:
                                    aligned_a.append(seq_a[i_tmp-1])
                                    aligned_b.append("-")
                                    i_tmp -= 1
                                else:
                                    # Check if this cell is a gap in seq_b
                                    if j_tmp > 1 and dp[i_tmp][j_tmp] == dp[i_tmp][j_tmp-1] + gap_extend:
                                        aligned_a.append("-")
                                        aligned_b.append(seq_b[j_tmp-1])
                                        j_tmp -= 1
                        else:
                            # If one of the indices is 0, handle the remaining sequence
                            if i_tmp > 0:
                                aligned_a.append(seq_a[i_tmp-1])
                                aligned_b.append("-")
                                i_tmp -= 1
                            else:
                                aligned_a.append("-")
                                aligned_b.append(seq_b[j_tmp-1])
                                j_tmp -= 1
                    aligned_a = aligned_a[::-1]
                    aligned_b = aligned_b[::-1]
                    return (max_score, "".join(aligned_a), "".join(aligned_b))
    
    # Handle global mode
    else:
        i, j = len_a, len_b
        aligned_a = []
        aligned_b = []
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                # Check if this cell is a match/mismatch
                if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                else:
                    # Check if this cell is a gap in seq_a
                    if i > 1 and dp[i][j] == dp[i-1][j] + gap_extend:
                        aligned_a.append(seq_a[i-1])
                        aligned_b.append("-")
                        i -= 1
                    else:
                        # Check if this cell is a gap in seq_b
                        if j > 1 and dp[i][j] == dp[i][j-1] + gap_extend:
                            aligned_a.append("-")
                            aligned_b.append(seq_b[j-1])
                            j -= 1
            else:
                # If one of the indices is 0, handle the remaining sequence
                if i > 0:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    i -= 1
                else:
                    aligned_a.append("-")
                    aligned_b.append(seq_b[j-1])
                    j -= 1
        aligned_a = aligned_a[::-1]
        aligned_b = aligned_b[::-1]
        return (dp[len_a][len_b], "".join(aligned_a), "".join(aligned_b))