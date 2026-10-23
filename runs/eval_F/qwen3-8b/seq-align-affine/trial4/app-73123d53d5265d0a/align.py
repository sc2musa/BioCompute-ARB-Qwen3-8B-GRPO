import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    # Initialize the score matrix
    m, n = len(seq_a), len(seq_b)
    score = np.zeros((m + 1, n + 1), dtype=int)
    # Initialize the traceback matrix
    traceback = np.zeros((m + 1, n + 1), dtype=int)
    # Fill the score matrix based on the mode
    if mode == "global":
        # Needleman-Wunsch algorithm
        # Initialize first row and column
        for i in range(m + 1):
            score[i][0] = i * gap_open
        for j in range(n + 1):
            score[0][j] = j * gap_open
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match_score = score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
                gap_a = score[i-1][j] + gap_extend
                gap_b = score[i][j-1] + gap_extend
                score[i][j] = max(match_score, gap_a, gap_b)
                # Traceback direction
                if match_score >= gap_a and match_score >= gap_b:
                    traceback[i][j] = 0  # Match
                elif gap_a >= gap_b:
                    traceback[i][j] = 1  # Gap in seq_b
                else:
                    traceback[i][j] = 2  # Gap in seq_a
    elif mode == "local":
        # Smith-Waterman algorithm
        # Initialize first row and column to 0
        for i in range(m + 1):
            score[i][0] = 0
        for j in range(n + 1):
            score[0][j] = 0
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match_score = score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
                gap_a = score[i-1][j] + gap_open
                gap_b = score[i][j-1] + gap_open
                score[i][j] = max(0, match_score, gap_a, gap_b)
                # Traceback direction
                if match_score >= gap_a and match_score >= gap_b:
                    traceback[i][j] = 0  # Match
                elif gap_a >= gap_b:
                    traceback[i][j] = 1  # Gap in seq_b
                else:
                    traceback[i][j] = 2  # Gap in seq_a
    else:
        raise ValueError("Invalid mode. Use 'global' or 'local'.")
    # Traceback to find the optimal alignment
    aligned_a = ""
    aligned_b = ""
    if mode == "global":
        i, j = m, n
        while i > 0 and j > 0:
            if traceback[i][j] == 0:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif traceback[i][j] == 1:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
            else:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
    elif mode == "local":
        # Find the maximum score in the matrix
        max_score = np.max(score)
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(score), score.shape)
        # Traceback from the maximum score
        aligned_a = ""
        aligned_b = ""
        while score[i][j] > 0:
            if traceback[i][j] == 0:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif traceback[i][j] == 1:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
            else:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
    # Return the score, aligned_a, aligned_b
    return (score[m][n], aligned_a, aligned_b)