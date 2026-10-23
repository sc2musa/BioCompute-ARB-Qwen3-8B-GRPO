import numpy as np

def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize DP table
    len_a = len(seq_a)
    len_b = len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize first row and column with gap penalties
    for i in range(1, len_a + 1):
        dp[i][0] = dp[i-1][0] + gap_open + (i-1)*gap_extend
    for j in range(1, len_b + 1):
        dp[0][j] = dp[0][j-1] + gap_open + (j-1)*gap_extend
    
    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate score for match/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Calculate score for gap in seq_a
            gap_a = dp[i-1][j] + gap_open + (i-1)*gap_extend
            
            # Calculate score for gap in seq_b
            gap_b = dp[i][j-1] + gap_open + (j-1)*gap_extend
            
            # Choose maximum score
            dp[i][j] = max(score + dp[i-1][j-1], gap_a, gap_b)
    
    # Handle local mode
    if mode == "local":
        # Find the maximum score in the DP table
        max_score = np.max(dp)
        # If the maximum score is negative, return (0, "", "")
        if max_score < 0:
            return (0, "", "")
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        # Initialize the DP table with the maximum score
        dp = np.full((len_a + 1, len_b + 1), max_score)
        # Fill the DP table with the original values
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if dp[i][j] != max_score:
                    dp[i][j] = 0
        # Fill the DP table again with the original values
        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                # Calculate score for match/mismatch
                if seq_a[i-1] == seq_b[j-1]:
                    score = match
                else:
                    score = mismatch
                
                # Calculate score for gap in seq_a
                gap_a = dp[i-1][j] + gap_open + (i-1)*gap_extend
                
                # Calculate score for gap in seq_b
                gap_b = dp[i][j-1] + gap_open + (j-1)*gap_extend
                
                # Choose maximum score
                dp[i][j] = max(score + dp[i-1][j-1], gap_a, gap_b)
    
    # Traceback to find optimal alignment
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
                # Check if we came from a match/mismatch
                if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                else:
                    # Check if we came from a gap in seq_a
                    if dp[i][j] == dp[i-1][j] + gap_open + (i-1)*gap_extend:
                        aligned_a.append(seq_a[i-1])
                        aligned_b.append("-")
                        i -= 1
                    else:
                        # Check if we came from a gap in seq_b
                        if dp[i][j] == dp[i][j-1] + gap_open + (j-1)*gap_extend:
                            aligned_a.append("-")
                            aligned_b.append(seq_b[j-1])
                            j -= 1
        else:
            # Handle remaining gaps
            if i > 0:
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
    return (dp[len_a][len_b], "".join(aligned_a), "".join(aligned_b))