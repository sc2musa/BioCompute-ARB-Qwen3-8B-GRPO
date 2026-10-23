import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize scoring matrix
    len_a, len_b = len(seq_a), len(seq_b)
    score_matrix = np.zeros((len_a + 1, len_b + 1))
    
    # Initialize first row and column based on mode
    if mode == "global":
        for i in range(1, len_a + 1):
            score_matrix[i][0] = score_matrix[i-1][0] + gap_open
        for j in range(1, len_b + 1):
            score_matrix[0][j] = score_matrix[0][j-1] + gap_open
    elif mode == "local":
        for i in range(1, len_a + 1):
            score_matrix[i][0] = 0
        for j in range(1, len_b + 1):
            score_matrix[0][j] = 0
    
    # Fill the scoring matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Score for match/mismatch
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Score for gap in seq_a
            gap_a_score = score_matrix[i-1][j] + gap_extend
            
            # Score for gap in seq_b
            gap_b_score = score_matrix[i][j-1] + gap_extend
            
            # Score for match/mismatch
            match_score = score_matrix[i-1][j-1] + match_score
            
            # Choose the maximum score
            score_matrix[i][j] = max(match_score, gap_a_score, gap_b_score)
    
    # Extract the maximum score
    if mode == "global":
        score = score_matrix[len_a][len_b]
    elif mode == "local":
        score = np.max(score_matrix)
    
    # Backtrack to find the optimal alignment
    aligned_a = ""
    aligned_b = ""
    i, j = len_a, len_b
    
    while i > 0 and j > 0:
        if mode == "local" and score_matrix[i][j] == 0:
            break
        
        if mode == "global" or (mode == "local" and score_matrix[i][j] > 0):
            # Check if the current cell is a match/mismatch
            if i > 0 and j > 0 and score_matrix[i][j] == score_matrix[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif i > 0 and score_matrix[i][j] == score_matrix[i-1][j] + gap_extend:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            elif j > 0 and score_matrix[i][j] == score_matrix[i][j-1] + gap_extend:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
        else:
            break
    
    return (score, aligned_a, aligned_b)