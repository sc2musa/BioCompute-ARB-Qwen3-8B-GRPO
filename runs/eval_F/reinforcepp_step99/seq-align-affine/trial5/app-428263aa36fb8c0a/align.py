import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the matrix
    len_a, len_b = len(seq_a), len(seq_b)
    matrix = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize the first row and column with gap penalties
    for i in range(1, len_a + 1):
        matrix[i, 0] = gap_open + (i-1)*gap_extend
    for j in range(1, len_b + 1):
        matrix[0, j] = gap_open + (j-1)*gap_extend
    
    # Fill the matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate the score for match/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Calculate the score for extending gaps
            gap_score = gap_extend
            
            # Choose the best score from the previous positions
            matrix[i, j] = max(
                matrix[i-1, j-1] + score,
                matrix[i-1, j] + gap_score,
                matrix[i, j-1] + gap_score
            )
    
    # Find the maximum score
    max_score = matrix[len_a, len_b]
    
    # Reconstruct the aligned sequences
    aligned_a = ""
    aligned_b = ""
    i, j = len_a, len_b
    while i > 0 and j > 0:
        if seq_a[i-1] == seq_b[j-1]:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        elif matrix[i-1, j] == matrix[i, j] - gap_extend:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
        elif matrix[i, j-1] == matrix[i, j] - gap_extend:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            j -= 1
        else:
            # This should not happen with proper implementation
            aligned_a = "-" + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
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