import numpy as np


def align(seq_a, seq_b, mode='global', match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    # Implement the alignment logic here
    # Using dynamic programming for pairwise alignment
    # Initialize the score matrix
    len_a = len(seq_a)
    len_b = len(seq_b)
    score_matrix = np.zeros((len_a + 1, len_b + 1))

    # Fill the score matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate the score for matching or mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            # Calculate the score for gap in seq_a
            gap_a = gap_open + (i - 1) * gap_extend
            # Calculate the score for gap in seq_b
            gap_b = gap_open + (j - 1) * gap_extend
            # Fill the score matrix with the maximum score
            score_matrix[i][j] = max(score_matrix[i-1][j-1] + score, score_matrix[i-1][j] + gap_a, score_matrix[i][j-1] + gap_b)

    # Extract the aligned sequences
    aligned_a = ""
    aligned_b = ""
    i = len_a
    j = len_b
    while i > 0 and j > 0:
        if seq_a[i-1] == seq_b[j-1]:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        elif score_matrix[i-1][j] > score_matrix[i][j-1]:
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
    # Return the score and aligned sequences
    return (score_matrix[len_a][len_b], aligned_a, aligned_b)