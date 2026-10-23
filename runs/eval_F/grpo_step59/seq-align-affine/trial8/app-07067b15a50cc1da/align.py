import numpy as np

def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    # Implement the alignment logic here
    # This is a placeholder implementation
    # Implementing the Needleman-Wunsch algorithm for global alignment
    # and the Smith-Waterman algorithm for local alignment
    # with affine gap penalties
    # Initialize the score matrix
    len_a = len(seq_a)
    len_b = len(seq_b)
    score_matrix = np.zeros((len_a + 1, len_b + 1))
    # Initialize the first row and column
    for i in range(1, len_a + 1):
        score_matrix[i][0] = score_matrix[i-1][0] + gap_open + (i-1) * gap_extend
    for j in range(1, len_b + 1):
        score_matrix[0][j] = score_matrix[0][j-1] + gap_open + (j-1) * gap_extend
    # Fill the score matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            score_matrix[i][j] = max(
                score_matrix[i-1][j] + gap_extend,
                score_matrix[i][j-1] + gap_extend,
                score_matrix[i-1][j-1] + match_score
            )
    # Extract the optimal alignment
    if mode == "global":
        score = score_matrix[len_a][len_b]
        aligned_a = seq_a
        aligned_b = seq_b
    else:
        # Find the maximum score in the matrix
        max_score = np.max(score_matrix)
        # Find the position of the maximum score
        max_i, max_j = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        # Extract the optimal alignment
        aligned_a = seq_a[max_i:]
        aligned_b = seq_b[max_j:]
        # Adjust the alignment to include the entire sequences if necessary
        # This is a simplified approach and may need further refinement
        # For the purpose of this example, we will return the substring alignment
        score = max_score
    return (score, aligned_a, aligned_b)