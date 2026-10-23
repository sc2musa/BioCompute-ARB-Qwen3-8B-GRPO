import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    if not seq_a and not seq_b:
        return (0, "", "")
    elif not seq_a:
        score = gap_open + (len(seq_b) - 1) * gap_extend
        if mode == "local" and score <= 0:
            return (0, "", "")
        else:
            return (score, "-" * len(seq_b), seq_b)
    elif not seq_b:
        score = gap_open + (len(seq_a) - 1) * gap_extend
        if mode == "local" and score <= 0:
            return (0, "", "")
        else:
            return (score, seq_a, "-" * len(seq_a))

    n, m = len(seq_a), len(seq_b)
    # Use large negative number instead of -inf to avoid overflow
    INF = -10**18
    M = np.full((n+1, m+1), INF, dtype=np.int64)
    X = np.full((n+1, m+1), INF, dtype=np.int64)
    Y = np.full((n+1, m+1), INF, dtype=np.int64)
    backtracking = np.full((n+1, m+1), "", dtype=object)

    if mode == "global":
        M[0][0] = 0
        for i in range(1, n+1):
            X[i][0] = gap_open + (i-1)*gap_extend
        for j in range(1, m+1):
            Y[0][j] = gap_open + (j-1)*gap_extend
    else:
        M[0][0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch

            # Compute M[i][j]
            if mode == "global":
                max_prev = max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1])
            else:
                max_prev = max(0, M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1])
            M[i][j] = score + max_prev
            if mode == "global":
                if M[i-1][j-1] == max_prev:
                    backtracking[i][j] = "M"
                elif X[i-1][j-1] == max_prev:
                    backtracking[i][j] = "X"
                else:
                    backtracking[i][j] = "Y"
            else:
                if M[i-1][j-1] == max_prev:
                    backtracking[i][j] = "M"
                elif X[i-1][j-1] == max_prev:
                    backtracking[i][j] = "X"
                elif Y[i-1][j-1] == max_prev:
                    backtracking[i][j] = "Y"
                else:
                    backtracking[i][j] = "M"

            # Compute X[i][j]
            option1 = M[i-1][j] + gap_open
            option2 = X[i-1][j] + gap_extend
            option3 = Y[i-1][j] + gap_open
            max_x = max(option1, option2, option3)
            X[i][j] = max_x
            if option1 == max_x:
                backtracking[i][j] = "X"
            elif option2 == max_x:
                backtracking[i][j] = "X"
            elif option3 == max_x:
                backtracking[i][j] = "X"

            # Compute Y[i][j]
            option1 = M[i][j-1] + gap_open
            option2 = Y[i][j-1] + gap_extend
            option3 = X[i][j-1] + gap_open
            max_y = max(option1, option2, option3)
            Y[i][j] = max_y
            if option1 == max_y:
                backtracking[i][j] = "Y"
            elif option2 == max_y:
                backtracking[i][j] = "Y"
            elif option3 == max_y:
                backtracking[i][j] = "Y"

    if mode == "global":
        best_score = max(M[n][m], X[n][m], Y[n][m])
        if best_score == M[n][m]:
            best_i, best_j = n, m
        elif best_score == X[n][m]:
            best_i, best_j = n, 0
        else:
            best_i, best_j = 0, m
    else:
        best_score = np.max(M)
        best_i, best_j = np.unravel_index(np.argmax(M), M.shape)
        if best_score <= 0:
            return (0, "", "")

    # Backtrack
    aligned_a = []
    aligned_b = []
    i, j = best_i, best_j
    while i > 0 or j > 0:
        direction = backtracking[i][j]
        if direction == "M":
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif direction == "X":
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        elif direction == "Y":
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
    aligned_a.reverse()
    aligned_b.reverse()
    return (best_score, "".join(aligned_a), "".join(aligned_b))