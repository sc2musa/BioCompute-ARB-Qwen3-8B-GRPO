import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the matrix
    len_a = len(seq_a)
    len_b = len(seq_b)
    matrix = np.zeros((len_a + 1, len_b + 1))

    # Initialize the first row and column with gap penalties
    for i in range(1, len_a + 1):
        matrix[i, 0] = matrix[i-1, 0] + gap_open + (i-1)*gap_extend
    for j in range(1, len_b + 1):
        matrix[0, j] = matrix[0, j-1] + gap_open + (j-1)*gap_extend

    # Fill the matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate the score for matching/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch

            # Calculate the maximum score
            matrix[i, j] = max(
                matrix[i-1, j] + score,
                matrix[i, j-1] + score,
                matrix[i-1, j-1] + score
            )

    # Traceback to find the optimal alignment
    i = len_a
    j = len_b
    aligned_a = []
    aligned_b = []

    while i > 0 and j > 0:
        if matrix[i, j] == matrix[i-1, j] + score:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        elif matrix[i, j] == matrix[i, j-1] + score:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
        else:
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1

    # Handle remaining characters
    while i > 0:
        aligned_a.append(seq_a[i-1])
        aligned_b.append("-")
        i -= 1
    while j > 0:
        aligned_a.append("-")
        aligned_b.append(seq_b[j-1])
        j -= 1

    # Return the result
    return (matrix[len_a, len_b], "".join(reversed(aligned_a)), "".join(reversed(aligned_b)))