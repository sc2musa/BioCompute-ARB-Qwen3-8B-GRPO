import math
import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """
    Return (score, aligned_a, aligned_b).
    """
    n = len(seq_a)
    m = len(seq_b)

    # Handle empty sequences
    if n == 0 and m == 0:
        return (0, "", "")

    # Initialize matrices
    M = [[-math.inf] * (m + 1) for _ in range(n + 1)]
    X = [[-math.inf] * (m + 1) for _ in range(n + 1)]
    Y = [[-math.inf] * (m + 1) for _ in range(n + 1)]

    # Set initial conditions based on mode
    if mode == "global":
        M[0][0] = 0
        for i in range(1, n + 1):
            X[i][0] = gap_open + (i - 1) * gap_extend
        for j in range(1, m + 1):
            Y[0][j] = gap_open + (j - 1) * gap_extend
    else:  # local mode
        M[0][0] = 0
        for i in range(1, n + 1):
            X[i][0] = gap_open + (i - 1) * gap_extend
        for j in range(1, m + 1):
            Y[0][j] = gap_open + (j - 1) * gap_extend

    # Fill the matrices
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a_char = seq_a[i - 1]
            b_char = seq_b[j - 1]
            score = match if a_char == b_char else mismatch

            # Compute M[i][j]
            if mode == "local":
                current_max = max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1], 0)
            else:
                current_max = max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1])
            M[i][j] = score + current_max

            # Compute X[i][j]
            option1 = M[i - 1][j] + gap_open
            option2 = X[i - 1][j] + gap_extend
            option3 = Y[i - 1][j] + gap_open
            X[i][j] = max(option1, option2, option3)

            # Compute Y[i][j]
            option1 = M[i][j - 1] + gap_open
            option2 = Y[i][j - 1] + gap_extend
            option3 = X[i][j - 1] + gap_open
            Y[i][j] = max(option1, option2, option3)

    # Determine the best score
    if mode == "global":
        best_score = max(M[n][m], X[n][m], Y[n][m])
    else:  # local mode
        best_score = max([max(row) for row in M])
        if best_score <= 0:
            return (0, "", "")

    # Traceback to find the alignment
    aligned_a = []
    aligned_b = []
    current_i, current_j = n, m
    if mode == "global":
        if M[n][m] == best_score:
            current_matrix = "M"
        elif X[n][m] == best_score:
            current_matrix = "X"
        else:
            current_matrix = "Y"
    else:  # local mode
        # Find the cell with the best score in M
        max_val = -math.inf
        for i in range(n + 1):
            for j in range(m + 1):
                if M[i][j] == best_score:
                    current_i, current_j = i, j
                    current_matrix = "M"
                    break
            else:
                continue
            break

    while current_i > 0 and current_j > 0:
        if current_matrix == "M":
            a_char = seq_a[current_i - 1]
            b_char = seq_b[current_j - 1]
            aligned_a.append(a_char)
            aligned_b.append(b_char)
            # Determine previous matrix
            s = match if a_char == b_char else mismatch
            prev_values = [M[current_i - 1][current_j - 1], X[current_i - 1][current_j - 1], Y[current_i - 1][current_j - 1]]
            max_prev = max(prev_values)
            if prev_values[0] == max_prev:
                current_matrix = "M"
            elif prev_values[1] == max_prev:
                current_matrix = "X"
            else:
                current_matrix = "Y"
            current_i -= 1
            current_j -= 1
        elif current_matrix == "X":
            a_char = seq_a[current_i - 1]
            aligned_a.append(a_char)
            aligned_b.append("-")
            # Determine previous matrix
            prev_values = [M[current_i - 1][current_j], X[current_i - 1][current_j], Y[current_i - 1][current_j]]
            max_prev = max(prev_values)
            if prev_values[0] == max_prev:
                current_matrix = "M"
            elif prev_values[1] == max_prev:
                current_matrix = "X"
            else:
                current_matrix = "Y"
            current_i -= 1
        elif current_matrix == "Y":
            b_char = seq_b[current_j - 1]
            aligned_a.append("-")
            aligned_b.append(b_char)
            # Determine previous matrix
            prev_values = [M[current_i][current_j - 1], X[current_i][current_j - 1], Y[current_i][current_j - 1]]
            max_prev = max(prev_values)
            if prev_values[0] == max_prev:
                current_matrix = "M"
            elif prev_values[1] == max_prev:
                current_matrix = "X"
            else:
                current_matrix = "Y"
            current_j -= 1

    # Add remaining characters
    while current_i > 0:
        aligned_a.append(seq_a[current_i - 1])
        aligned_b.append("-")
        current_i -= 1
    while current_j > 0:
        aligned_a.append("-")
        aligned_b.append(seq_b[current_j - 1])
        current_j -= 1

    # Reverse to get correct order
    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]

    return (best_score, "".join(aligned_a), "".join(aligned_b))
