import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the matrix
    len_a = len(seq_a)
    len_b = len(seq_b)
    
    # Create matrices for global and local modes
    matrix_global = np.zeros((len_a + 1, len_b + 1), dtype=int)
    matrix_local = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Fill the matrices
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate the score for matching/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Calculate the best score for global mode
            up = matrix_global[i-1, j] + gap_open
            left = matrix_global[i, j-1] + gap_open
            diag = matrix_global[i-1, j-1] + score
            matrix_global[i, j] = max(up, left, diag)
            
            # Calculate the best score for local mode
            up_local = matrix_local[i-1, j] + gap_open
            left_local = matrix_local[i, j-1] + gap_open
            diag_local = matrix_local[i-1, j-1] + score
            matrix_local[i, j] = max(up_local, left_local, diag_local, 0)
    
    # Determine the best score based on mode
    if mode == "global":
        best_score = matrix_global[len_a, len_b]
    else:
        best_score = matrix_local[len_a, len_b]
    
    # Reconstruct the alignment
    aligned_a = ""
    aligned_b = ""
    i, j = len_a, len_b
    while i > 0 and j > 0:
        # Check if we're at the end of the sequences
        if i == 0 or j == 0:
            break
        
        # Check if the current characters match
        if seq_a[i-1] == seq_b[j-1]:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        else:
            # Check if we should extend a gap
            if mode == "global":
                if matrix_global[i-1, j] == matrix_global[i, j] - gap_open:
                    aligned_a = "-" + aligned_a
                    aligned_b = seq_b[j-1] + aligned_b
                    j -= 1
                elif matrix_global[i, j-1] == matrix_global[i, j] - gap_open:
                    aligned_a = seq_a[i-1] + aligned_a
                    aligned_b = "-" + aligned_b
                    i -= 1
                else:
                    # We're at a diagonal match
                    aligned_a = seq_a[i-1] + aligned_a
                    aligned_b = seq_b[j-1] + aligned_b
                    i -= 1
                    j -= 1
            else:
                # Local mode handling
                if matrix_local[i-1, j] == matrix_local[i, j] - gap_open:
                    aligned_a = "-" + aligned_a
                    aligned_b = seq_b[j-1] + aligned_b
                    j -= 1
                elif matrix_local[i, j-1] == matrix_local[i, j] - gap_open:
                    aligned_a = seq_a[i-1] + aligned_a
                    aligned_b = "-" + aligned_b
                    i -= 1
                else:
                    # We're at a diagonal match
                    aligned_a = seq_a[i-1] + aligned_a
                    aligned_b = seq_b[j-1] + aligned_b
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
    
    return (best_score, aligned_a, aligned_b)