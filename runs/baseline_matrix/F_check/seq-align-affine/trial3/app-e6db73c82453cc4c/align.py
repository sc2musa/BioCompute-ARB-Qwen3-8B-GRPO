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
    
    # Initialize DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize first row and column for global alignment
    if mode == "global":
        for i in range(len_a + 1):
            dp[i][0] = i * gap_open + (i - 1) * gap_extend
        for j in range(len_b + 1):
            dp[0][j] = j * gap_open + (j - 1) * gap_extend
    
    # Initialize first row and column for local alignment
    elif mode == "local":
        for i in range(len_a + 1):
            dp[i][0] = 0
        for j in range(len_b + 1):
            dp[0][j] = 0
    
    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Score for match/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Score for diagonal (match/mismatch)
            diag_score = dp[i-1][j-1] + score
            
            # Score for up (gap in seq_b)
            up_score = dp[i-1][j] + gap_extend
            
            # Score for left (gap in seq_a)
            left_score = dp[i][j-1] + gap_extend
            
            # Choose the best score
            if mode == "global":
                dp[i][j] = max(diag_score, up_score, left_score)
            elif mode == "local":
                dp[i][j] = max(diag_score, up_score, left_score, 0)
    
    # Traceback to find the optimal alignment
    if mode == "global":
        i, j = len_a, len_b
    else:
        # Find the maximum score in the DP table for local alignment
        max_score = dp.max()
        if max_score <= 0:
            return (0, "", "")
        # Find the position of the maximum score
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if dp[i][j] == max_score:
                    break
            else:
                continue
            break
    
    # Backtrack to find the alignment
    aligned_a = []
    aligned_b = []
    while i > 0 and j > 0:
        if seq_a[i-1] == seq_b[j-1]:
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        else:
            # Check if we came from diagonal (match/mismatch)
            if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            # Check if we came from up (gap in seq_b)
            elif dp[i][j] == dp[i-1][j] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            # Check if we came from left (gap in seq_a)
            elif dp[i][j] == dp[i][j-1] + gap_extend:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
    
    # Reverse the alignment to get the correct order
    aligned_a = ''.join(reversed(aligned_a))
    aligned_b = ''.join(reversed(aligned_b))
    
    # Handle the remaining characters for global alignment
    if mode == "global":
        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        while j > 0:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
    
    # Handle the remaining characters for local alignment
    elif mode == "local":
        # Remove the trailing gaps from the alignment
        while aligned_a and aligned_a[-1] == "-":
            aligned_a.pop()
            aligned_b.pop()
        while aligned_b and aligned_b[-1] == "-":
            aligned_a.pop()
            aligned_b.pop()
    
    # Calculate the score
    score = 0
    for a, b in zip(aligned_a, aligned_b):
        if a == "-" or b == "-":
            if a == "-" and b == "-":
                continue
            # Check if it's a gap in seq_a or seq_b
            if a == "-":
                # Gap in seq_a
                if len(aligned_a) > 1 and aligned_a[-2] == "-":
                    # Extend a gap
                    score += gap_extend
                else:
                    # Open a gap
                    score += gap_open
            else:
                # Gap in seq_b
                if len(aligned_b) > 1 and aligned_b[-2] == "-":
                    # Extend a gap
                    score += gap_extend
                else:
                    # Open a gap
                    score += gap_open
        else:
            if a == b:
                score += match
            else:
                score += mismatch
    
    return (score, aligned_a, aligned_b)