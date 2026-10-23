import numpy as np

def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    # Implement the alignment algorithm here
    # Create the scoring matrix
    len_a, len_b = len(seq_a), len(seq_b)
    score_matrix = np.zeros((len_a + 1, len_b + 1), dtype=int)

    # Initialize the matrix
    if mode == "global":
        score_matrix[0, 0] = 0
        for i in range(1, len_a + 1):
            score_matrix[i, 0] = score_matrix[i-1, 0] + gap_open
        for j in range(1, len_b + 1):
            score_matrix[0, j] = score_matrix[0, j-1] + gap_open
    else:  # local
        score_matrix[0, 0] = 0
        for i in range(1, len_a + 1):
            score_matrix[i, 0] = -1000000  # Use a large negative number instead of -inf
        for j in range(1, len_b + 1):
            score_matrix[0, j] = -1000000

    # Fill the matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            score_matrix[i, j] = max(
                score_matrix[i-1, j] + gap_extend,
                score_matrix[i, j-1] + gap_extend,
                score_matrix[i-1, j-1] + match_score
            )

    # Find the optimal score
    if mode == "global":
        optimal_score = score_matrix[len_a, len_b]
    else:  # local
        optimal_score = np.max(score_matrix)

    # Traceback to find the alignment
    aligned_a = ""
    aligned_b = ""
    i, j = len_a, len_b
    while i > 0 or j > 0:
        if mode == "global":
            if i > 0 and j > 0:
                if score_matrix[i, j] == score_matrix[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                    aligned_a = seq_a[i-1] + aligned_a
                    aligned_b = seq_b[j-1] + aligned_b
                    i -= 1
                    j -= 1
                elif i > 0 and score_matrix[i, j] == score_matrix[i-1, j] + gap_extend:
                    aligned_a = seq_a[i-1] + aligned_a
                    aligned_b = "-" + aligned_b
                    i -= 1
                else:
                    aligned_a = "-" + aligned_a
                    aligned_b = seq_b[j-1] + aligned_b
                    j -= 1
            else:
                if i > 0:
                    aligned_a = seq_a[i-1] + aligned_a
                    aligned_b = "-" + aligned_b
                    i -= 1
                else:
                    aligned_a = "-" + aligned_a
                    aligned_b = seq_b[j-1] + aligned_b
                    j -= 1
        else:  # local
            if i > 0 and j > 0:
                if score_matrix[i, j] == score_matrix[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                    aligned_a = seq_a[i-1] + aligned_a
                    aligned_b = seq_b[j-1] + aligned_b
                    i -= 1
                    j -= 1
                elif i > 0 and score_matrix[i, j] == score_matrix[i-1, j] + gap_extend:
                    aligned_a = seq_a[i-1] + aligned_a
                    aligned_b = "-" + aligned_b
                    i -= 1
                else:
                    aligned_a = "-" + aligned_a
                    aligned_b = seq_b[j-1] + aligned_b
                    j -= 1
            else:
                if i > 0:
                    aligned_a = seq_a[i-1] + aligned_a
                    aligned_b = "-" + aligned_b
                    i -= 1
                else:
                    aligned_a = "-" + aligned_a
                    aligned_b = seq_b[j-1] + aligned_b
                    j -= 1

    return (optimal_score, aligned_a, aligned_b)