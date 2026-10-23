import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the scoring matrix
    len_a, len_b = len(seq_a), len(seq_b)
    score_matrix = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize the first row and column based on the mode
    if mode == "global":
        # Needleman-Wunsch style
        for i in range(1, len_a + 1):
            score_matrix[i][0] = score_matrix[i-1][0] + gap_open
        for j in range(1, len_b + 1):
            score_matrix[0][j] = score_matrix[0][j-1] + gap_open
    else:
        # Smith-Waterman style
        for i in range(len_a + 1):
            score_matrix[i][0] = 0
        for j in range(len_b + 1):
            score_matrix[0][j] = 0
    
    # Fill the scoring matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Score for match/mismatch
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Score for gap in seq_a
            gap_a = score_matrix[i-1][j] + gap_open
            
            # Score for gap in seq_b
            gap_b = score_matrix[i][j-1] + gap_open
            
            # Score for extending a gap in seq_a
            gap_a_extend = score_matrix[i-1][j] + gap_extend
            
            # Score for extending a gap in seq_b
            gap_b_extend = score_matrix[i][j-1] + gap_extend
            
            # Choose the maximum score
            score_matrix[i][j] = max(
                match_score + score_matrix[i-1][j-1],
                gap_a,
                gap_b,
                gap_a_extend,
                gap_b_extend
            )
    
    # Traceback to find the optimal alignment
    if mode == "global":
        # Needleman-Wunsch style
        i, j = len_a, len_b
        aligned_a, aligned_b = [], []
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif score_matrix[i][j] == score_matrix[i-1][j-1] + match if seq_a[i-1] == seq_b[j-1] else score_matrix[i-1][j-1] + mismatch:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif score_matrix[i][j] == score_matrix[i-1][j] + gap_open:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            elif score_matrix[i][j] == score_matrix[i][j-1] + gap_open:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
            elif score_matrix[i][j] == score_matrix[i-1][j] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            elif score_matrix[i][j] == score_matrix[i][j-1] + gap_extend:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
        
        # Add remaining characters
        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        while j > 0:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
        
        # Reverse the aligned sequences
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
        
        # Calculate the score
        score = score_matrix[len_a][len_b]
    else:
        # Smith-Waterman style
        # Find the maximum score in the matrix
        max_score = np.max(score_matrix)
        if max_score <= 0:
            return (0, "", "")
        
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        
        # Traceback to find the optimal alignment
        aligned_a, aligned_b = [], []
        while i > 0 and j > 0 and score_matrix[i][j] > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif score_matrix[i][j] == score_matrix[i-1][j-1] + match if seq_a[i-1] == seq_b[j-1] else score_matrix[i-1][j-1] + mismatch:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif score_matrix[i][j] == score_matrix[i-1][j] + gap_open:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            elif score_matrix[i][j] == score_matrix[i][j-1] + gap_open:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
            elif score_matrix[i][j] == score_matrix[i-1][j] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            elif score_matrix[i][j] == score_matrix[i][j-1] + gap_extend:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
        
        # Reverse the aligned sequences
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
        
        # Calculate the score
        score = max_score
    
    return (score, aligned_a, aligned_b)
