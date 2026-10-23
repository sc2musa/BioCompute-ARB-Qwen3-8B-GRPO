import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Handle empty sequences
    if not seq_a and not seq_b:
        return (0, "", "")
    elif not seq_a:
        return (0, "", seq_b)
    elif not seq_b:
        return (0, seq_a, "")

    # Initialize scoring matrix
    len_a, len_b = len(seq_a), len(seq_b)
    score = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize first row and column
    if mode == "global":
        # Global alignment (Needleman-Wunsch)
        for i in range(1, len_a + 1):
            score[i][0] = score[i-1][0] + gap_open
        for j in range(1, len_b + 1):
            score[0][j] = score[0][j-1] + gap_open
    else:
        # Local alignment (Smith-Waterman)
        for i in range(1, len_a + 1):
            score[i][0] = 0
        for j in range(1, len_b + 1):
            score[0][j] = 0

    # Fill the scoring matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Match/mismatch score
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Gap in seq_a
            gap_a = score[i-1][j] + gap_extend
            
            # Gap in seq_b
            gap_b = score[i][j-1] + gap_extend
            
            # Diagonal score
            diag_score = score[i-1][j-1] + match_score
            
            # Choose the maximum score
            if mode == "global":
                score[i][j] = max(diag_score, gap_a, gap_b)
            else:
                score[i][j] = max(diag_score, gap_a, gap_b, 0)

    # Traceback to find the optimal alignment
    if mode == "global":
        # Global alignment
        i, j = len_a, len_b
        aligned_a, aligned_b = ",", ","
        
        while i > 0 and j > 0:
            if score[i][j] == score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif score[i][j] == score[i-1][j] + gap_extend:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            else:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
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
        
        score = score[len_a][len_b]
    else:
        # Local alignment
        # Find the maximum score in the matrix
        max_score = np.max(score)
        if max_score <= 0:
            return (0, "", "")
        
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(score), score.shape)
        
        # Traceback to find the optimal alignment
        aligned_a, aligned_b = ",", ","
        
        while score[i][j] > 0:
            if score[i][j] == score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif score[i][j] == score[i-1][j] + gap_extend:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            else:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
        
        score = max_score

    return (score, aligned_a, aligned_b)