import numpy as np

def align(seq_a, seq_b, mode='global', match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    # Implement the align function here
    # Initialize the score matrix
    len_a, len_b = len(seq_a), len(seq_b)
    score = np.zeros((len_a + 1, len_b + 1))
    # Fill the score matrix
    for i in range(len_a + 1):
        for j in range(len_b + 1):
            if i == 0 and j == 0:
                score[i][j] = 0
            elif i == 0:
                # Handle gaps in seq_a
                score[i][j] = score[i][j-1] + gap_open if j == 1 else score[i][j-1] + gap_extend
            elif j == 0:
                # Handle gaps in seq_b
                score[i][j] = score[i-1][j] + gap_open if i == 1 else score[i-1][j] + gap_extend
            else:
                # Calculate the score for matching or mismatching characters
                if seq_a[i-1] == seq_b[j-1]:
                    score[i][j] = score[i-1][j-1] + match
                else:
                    score[i][j] = score[i-1][j-1] + mismatch
    # Find the optimal alignment
    if mode == 'global':
        max_score = score[len_a][len_b]
        aligned_a = seq_a
        aligned_b = seq_b
    else:
        # Find the maximum score in the matrix
        max_score = np.max(score)
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(score), score.shape)
        # Extract the aligned sequences
        aligned_a = seq_a[i:j]
        aligned_b = seq_b[i:j]
    return (max_score, aligned_a, aligned_b)
