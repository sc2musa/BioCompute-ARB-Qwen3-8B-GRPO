import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize matrix
    len_a = len(seq_a)
    len_b = len(seq_b)
    matrix = np.zeros((len_a + 1, len_b + 1))
    
    # Fill matrix
    for i in range(len_a + 1):
        for j in range(len_b + 1):
            if i == 0 and j == 0:
                matrix[i, j] = 0
            elif i == 0:
                # Gap in seq_a
                matrix[i, j] = matrix[i, j-1] + gap_open + (j-1)*gap_extend
            elif j == 0:
                # Gap in seq_b
                matrix[i, j] = matrix[i-1, j] + gap_open + (i-1)*gap_extend
            else:
                # Calculate score for match/mismatch
                if seq_a[i-1] == seq_b[j-1]:
                    score = match
                else:
                    score = mismatch
                
                # Calculate score for gap in seq_a
                gap_a = matrix[i-1, j] + gap_open + (i-1)*gap_extend
                
                # Calculate score for gap in seq_b
                gap_b = matrix[i, j-1] + gap_open + (j-1)*gap_extend
                
                # Choose maximum score
                matrix[i, j] = max(score + matrix[i-1, j-1], gap_a, gap_b)
    
    # Find optimal alignment
    if mode == "global":
        # Find maximum score in the matrix
        max_score = np.max(matrix)
        
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
        
        # Backtrack to find the alignment
        aligned_a = ""
        aligned_b = ""
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif matrix[i, j] == matrix[i-1, j] + gap_open + (i-1)*gap_extend:
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
        
        return (max_score, aligned_a, aligned_b)
    else:
        # Local alignment
        # Find the maximum score in the matrix
        max_score = np.max(matrix)
        
        # If no positive score, return (0, "", "")
        if max_score <= 0:
            return (0, "", "")
        
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
        
        # Backtrack to find the alignment
        aligned_a = ""
        aligned_b = ""
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif matrix[i, j] == matrix[i-1, j] + gap_open + (i-1)*gap_extend:
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
        
        return (max_score, aligned_a, aligned_b)