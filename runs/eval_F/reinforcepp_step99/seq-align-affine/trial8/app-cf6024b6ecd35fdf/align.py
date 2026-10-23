import numpy as np

def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the matrix
    len_a, len_b = len(seq_a), len(seq_b)
    matrix = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Fill the matrix
    for i in range(len_a + 1):
        for j in range(len_b + 1):
            if i == 0 and j == 0:
                matrix[i, j] = 0
            elif i == 0:
                matrix[i, j] = gap_open + (j - 1) * gap_extend
            elif j == 0:
                matrix[i, j] = gap_open + (i - 1) * gap_extend
            else:
                # Calculate the score for matching/mismatch
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                # Calculate the score for gap in seq_a
                gap_a = matrix[i-1, j] + gap_open + (i-1 - 1) * gap_extend
                # Calculate the score for gap in seq_b
                gap_b = matrix[i, j-1] + gap_open + (j-1 - 1) * gap_extend
                # Take the maximum of the three options
                matrix[i, j] = max(gap_a, gap_b, matrix[i-1, j-1] + score)
    
    # Backtrack to find the optimal alignment
    i, j = len_a, len_b
    aligned_a, aligned_b = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            else:
                # Choose the best option based on the matrix
                if matrix[i-1, j] + gap_open + (i-1 - 1) * gap_extend >= matrix[i, j-1] + gap_open + (j-1 - 1) * gap_extend:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    i -= 1
                else:
                    aligned_a.append("-")
                    aligned_b.append(seq_b[j-1])
                    j -= 1
        elif i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        else:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
    
    # Reverse the alignments
    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]
    
    # Return the score and alignments
    return (matrix[len_a, len_b], "".join(aligned_a), "".join(aligned_b))