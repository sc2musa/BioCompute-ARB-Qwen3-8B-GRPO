import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Create matrices
    len_a, len_b = len(seq_a), len(seq_b)
    score_matrix = np.zeros((len_a + 1, len_b + 1))
    
    # Initialize first row and column
    for i in range(len_a + 1):
        score_matrix[i, 0] = i * gap_open
    for j in range(len_b + 1):
        score_matrix[0, j] = j * gap_open
    
    # Fill the matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate the score for this position
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Calculate the score for the three possible operations
            score_matrix[i, j] = max(
                score_matrix[i-1, j-1] + score,  # Match/replace
                score_matrix[i-1, j] + gap_extend,  # Gap in seq_b
                score_matrix[i, j-1] + gap_extend   # Gap in seq_a
            )
    
    # Find the optimal alignment
    if mode == "global":
        # Find the maximum score in the last row and column
        max_score = max(score_matrix[-1, :], score_matrix[:, -1])
        # Find the position of the maximum score
        max_pos = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        
        # Backtrack to find the alignment
        aligned_a, aligned_b = "", ""
        i, j = max_pos
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif score_matrix[i, j] == score_matrix[i-1, j] + gap_extend:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
            else:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
        
        # Add the remaining characters
        while i > 0:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        while j > 0:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
    else:
        # Find the maximum score in the matrix
        max_score = np.max(score_matrix)
        # Find the position of the maximum score
        max_pos = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        
        # Backtrack to find the alignment
        aligned_a, aligned_b = "", ""
        i, j = max_pos
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif score_matrix[i, j] == score_matrix[i-1, j] + gap_extend:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
            else:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
        
        # Add the remaining characters
        while i > 0:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        while j > 0:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
    
    return (max_score, aligned_a, aligned_b)