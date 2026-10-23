import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)
    
    # Initialize DP table with three states: 0 (match/mismatch), 1 (gap in seq_a), 2 (gap in seq_b)
    dp = np.full((len_a + 1, len_b + 1, 3), -np.inf, dtype=int)
    dp[0][0][0] = 0
    
    # Fill DP table
    for i in range(len_a + 1):
        for j in range(len_b + 1):
            # State 0: match/mismatch
            if i > 0 and j > 0:
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                dp[i][j][0] = max(
                    dp[i-1][j-1][0] + score,
                    dp[i-1][j-1][1] + score,
                    dp[i-1][j-1][2] + score
                )
            
            # State 1: gap in seq_a
            if i > 0:
                dp[i][j][1] = max(
                    dp[i-1][j][1] + gap_extend,
                    dp[i-1][j][0] + gap_open
                )
            
            # State 2: gap in seq_b
            if j > 0:
                dp[i][j][2] = max(
                    dp[i][j-1][2] + gap_extend,
                    dp[i][j-1][0] + gap_open
                )
    
    # Determine the score based on mode
    if mode == "global":
        score = dp[len_a][len_b][0]
        # Backtrack to find the alignment
        aligned_a, aligned_b = backtrack_global(dp, seq_a, seq_b, len_a, len_b, gap_open, gap_extend)
    else:  # mode == "local"
        # Find the maximum score in the DP table
        max_score = np.max(dp)
        if max_score <= 0:
            return (0, "", "")
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        # Backtrack to find the alignment
        aligned_a, aligned_b = backtrack_local(dp, seq_a, seq_b, i, j, gap_open, gap_extend)
        score = max_score
    
    return (score, aligned_a, aligned_b)


def backtrack_global(dp, seq_a, seq_b, len_a, len_b, gap_open, gap_extend):
    aligned_a = []
    aligned_b = []
    i, j = len_a, len_b
    
    while i > 0 and j > 0:
        if dp[i][j][0] == dp[i-1][j-1][0] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif dp[i][j][0] == dp[i-1][j-1][1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif dp[i][j][0] == dp[i-1][j-1][2] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        else:
            # Handle gaps
            if dp[i][j][1] == dp[i-1][j][1] + gap_extend:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif dp[i][j][1] == dp[i-1][j][0] + gap_open:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif dp[i][j][2] == dp[i][j-1][2] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                j -= 1
            elif dp[i][j][2] == dp[i][j-1][0] + gap_open:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                j -= 1
    
    # Handle remaining characters
    while i > 0:
        aligned_a.append(seq_a[i-1])
        aligned_b.append("-")
        i -= 1
    while j > 0:
        aligned_a.append("-")
        aligned_b.append(seq_b[j-1])
        j -= 1
    
    return ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b))

def backtrack_local(dp, seq_a, seq_b, i, j, gap_open, gap_extend):
    aligned_a = []
    aligned_b = []
    
    while i > 0 and j > 0:
        if dp[i][j][0] == dp[i-1][j-1][0] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif dp[i][j][0] == dp[i-1][j-1][1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif dp[i][j][0] == dp[i-1][j-1][2] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        else:
            # Handle gaps
            if dp[i][j][1] == dp[i-1][j][1] + gap_extend:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif dp[i][j][1] == dp[i-1][j][0] + gap_open:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif dp[i][j][2] == dp[i][j-1][2] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                j -= 1
            elif dp[i][j][2] == dp[i][j-1][0] + gap_open:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                j -= 1
    
    # Handle remaining characters
    while i > 0:
        aligned_a.append(seq_a[i-1])
        aligned_b.append("-")
        i -= 1
    while j > 0:
        aligned_a.append("-")
        aligned_b.append(seq_b[j-1])
        j -= 1
    
    return ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b))
