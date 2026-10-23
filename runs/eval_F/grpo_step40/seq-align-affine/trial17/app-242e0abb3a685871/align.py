import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize matrices
    len_a = len(seq_a)
    len_b = len(seq_b)
    
    # Create scoring matrix
    score_matrix = np.zeros((len_a + 1, len_b + 1))
    
    # Fill the matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate score for match/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Calculate gap penalties
            gap_score = gap_open + (i-1) * gap_extend if i > 0 else 0
            gap_score2 = gap_open + (j-1) * gap_extend if j > 0 else 0
            
            # Calculate the maximum score
            score_matrix[i, j] = max(
                score_matrix[i-1, j-1] + score,
                score_matrix[i-1, j] + gap_score,
                score_matrix[i, j-1] + gap_score2
            )
    
    # Extract the optimal alignment
    if mode == "global":
        # Find the optimal score
        optimal_score = score_matrix[len_a, len_b]
        
        # Backtrack to find the alignment
        aligned_a = ""
        aligned_b = ""
        i, j = len_a, len_b
        while i > 0 or j > 0:
            if i > 0 and j > 0 and seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif i > 0:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            elif j > 0:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
            else:
                break
        
        return (optimal_score, aligned_a, aligned_b)
    else:
        # Find the maximum score in the matrix
        max_score = np.max(score_matrix)
        
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        
        # Backtrack to find the alignment
        aligned_a = ""
        aligned_b = ""
        while i > 0 or j > 0:
            if i > 0 and j > 0 and seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif i > 0:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            elif j > 0:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
            else:
                break
        
        return (max_score, aligned_a, aligned_b)