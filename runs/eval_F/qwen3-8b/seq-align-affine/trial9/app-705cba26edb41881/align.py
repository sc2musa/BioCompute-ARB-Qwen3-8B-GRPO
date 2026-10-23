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
    
    # Initialize scoring matrix
    m, n = len(seq_a), len(seq_b)
    score = np.zeros((m + 1, n + 1), dtype=int)
    
    # Initialize first row and column based on mode
    if mode == "global":
        # Needleman-Wunsch style
        for i in range(1, m + 1):
            score[i][0] = score[i-1][0] + gap_open
        for j in range(1, n + 1):
            score[0][j] = score[0][j-1] + gap_open
    else:
        # Smith-Waterman style
        # Initialize with 0s for local alignment
        for i in range(m + 1):
            for j in range(n + 1):
                score[i][j] = 0
    
    # Fill the scoring matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Score for match/mismatch
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Score for gap in seq_a
            gap_a = score[i-1][j] + gap_extend
            
            # Score for gap in seq_b
            gap_b = score[i][j-1] + gap_extend
            
            # Score for extending existing gaps
            score[i][j] = max(
                score[i-1][j-1] + match_score,
                gap_a,
                gap_b
            )
    
    # Determine the optimal score based on mode
    if mode == "global":
        score_val = score[m][n]
    else:
        # Find the maximum score in the matrix for local alignment
        score_val = np.max(score)
        # Find the position of the maximum score
        max_i, max_j = np.unravel_index(np.argmax(score), score.shape)
        
    # Backtrack to find the alignment
    aligned_a = ""
    aligned_b = ""
    
    if mode == "global":
        i, j = m, n
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif score[i][j] == score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif score[i][j] == score[i-1][j] + gap_open:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            else:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
    else:
        # Local alignment backtracking
        i, j = max_i, max_j
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif score[i][j] == score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif score[i][j] == score[i-1][j] + gap_open:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            else:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
    
    # Check if the score is positive for local alignment
    if mode == "local" and score_val <= 0:
        return (0, "", "")
    
    return (score_val, aligned_a, aligned_b)