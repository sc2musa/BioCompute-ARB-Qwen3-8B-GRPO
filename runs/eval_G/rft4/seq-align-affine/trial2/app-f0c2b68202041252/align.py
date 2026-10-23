import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    len_a = len(seq_a)
    len_b = len(seq_b)
    INF = float('-inf')

    # Initialize DP tables
    M = [[INF]*(len_b+1) for _ in range(len_a+1)]
    X = [[INF]*(len_b+1) for _ in range(len_a+1)]
    Y = [[INF]*(len_b+1) for _ in range(len_a+1)]
    S = [[INF]*(len_b+1) for _ in range(len_a+1)] if mode == "local" else None

    # Base cases for global mode
    M[0][0] = 0
    for j in range(1, len_b+1):
        X[0][j] = gap_open + (j-1)*gap_extend
    for i in range(1, len_a+1):
        Y[i][0] = gap_open + (i-1)*gap_extend

    for i in range(1, len_a+1):
        for j in range(1, len_b+1):
            # Compute score for current column
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch

            # Compute M[i][j]
            M[i][j] = max(
                M[i-1][j-1] + score,
                X[i-1][j] + score,
                Y[i][j-1] + score
            )

            # Compute X[i][j]
            X[i][j] = max(
                X[i-1][j] + gap_extend,
                M[i-1][j] + gap_open
            )

            # Compute Y[i][j]
            Y[i][j] = max(
                Y[i][j-1] + gap_extend,
                M[i-1][j] + gap_open
            )

    # Local mode
    if mode == "local":
        S[0][0] = 0
        for i in range(1, len_a+1):
            for j in range(1, len_b+1):
                if seq_a[i-1] == seq_b[j-1]:
                    score = match
                else:
                    score = mismatch
                S[i][j] = max(
                    S[i-1][j-1] + score,
                    S[i-1][j],
                    S[i][j-1],
                    0
                )
        best_score = max(max(row) for row in S)
        # Find the best cell
        best_i, best_j = 0, 0
        for i in range(len_a+1):
            for j in range(len_b+1):
                if S[i][j] > S[best_i][best_j]:
                    best_i, best_j = i, j
        # Backtrack from best_i, best_j
        aligned_a, aligned_b = "", ""
        i, j = best_i, best_j
        while i > 0 and j > 0:
            if S[i][j] == S[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif S[i][j] == S[i-1][j]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            elif S[i][j] == S[i][j-1]:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
            # else: invalid case (double gap) - should not happen
        # Handle edge cases
        while i > 0:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        while j > 0:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
        return (best_score, aligned_a, aligned_b) if best_score > 0 else (0, "", "")
    else:
        best_score = M[len_a][len_b]
        # Backtrack from M[len_a][len_b]
        aligned_a, aligned_b = "", ""
        i, j = len_a, len_b
        while i > 0 and j > 0:
            if M[i][j] == M[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif M[i][j] == X[i-1][j] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
                j -= 1
            elif M[i][j] == Y[i][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            # else: invalid case (double gap) - should not happen
        # Handle edge cases
        while i > 0:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        while j > 0:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
        return (best_score, aligned_a, aligned_b)