def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    import numpy as np
    n = len(seq_a)
    m = len(seq_b)
    if n == 0 and m == 0:
        return (0, "", "")
    if n == 0:
        return (0, "-"*m, seq_b)
    if m == 0:
        return (0, seq_a, "-"*n)
    # Initialize matrices
    M = np.full((n+1, m+1), -np.inf)
    X = np.full((n+1, m+1), -np.inf)
    Y = np.full((n+1, m+1), -np.inf)
    back_M = np.full((n+1, m+1), None)
    back_X = np.full((n+1, m+1), None)
    back_Y = np.full((n+1, m+1), None)
    if mode == 'global':
        M[0][0] = 0
        for i in range(1, n+1):
            X[i][0] = gap_open + (i-1)*gap_extend
        for j in range(1, m+1):
            Y[0][j] = gap_open + (j-1)*gap_extend
    elif mode == 'local':
        M[0][0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            if seq_a[i-1] == seq_b[j-1]:
                s = match
            else:
                s = mismatch
            # Compute M[i][j]
            if mode == 'local':
                max_prev = max(0, M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1])
            else:
                max_prev = max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1])
            M[i][j] = s + max_prev
            if M[i-1][j-1] == max_prev:
                back_M[i][j] = 'M'
            elif X[i-1][j-1] == max_prev:
                back_M[i][j] = 'X'
            else:
                back_M[i][j] = 'Y'
            # Compute X[i][j]
            option1 = X[i-1][j] + gap_extend
            option2 = M[i-1][j] + gap_open
            option3 = Y[i-1][j] + gap_open
            max_X = max(option1, option2, option3)
            X[i][j] = max_X
            if option1 == max_X:
                back_X[i][j] = 'X'
            elif option2 == max_X:
                back_X[i][j] = 'M'
            else:
                back_X[i][j] = 'Y'
            # Compute Y[i][j]
            option1 = Y[i][j-1] + gap_extend
            option2 = M[i][j-1] + gap_open
            option3 = X[i][j-1] + gap_open
            max_Y = max(option1, option2, option3)
            Y[i][j] = max_Y
            if option1 == max_Y:
                back_Y[i][j] = 'Y'
            elif option2 == max_Y:
                back_Y[i][j] = 'M'
            else:
                back_Y[i][j] = 'X'
    # Determine best score and backtracking
    if mode == 'global':
        best_score = max(M[n][m], X[n][m], Y[n][m])
        if best_score < 0:
            return (0, "", "")
        if M[n][m] == best_score:
            current_matrix = 'M'
            i, j = n, m
        elif X[n][m] == best_score:
            current_matrix = 'X'
            i, j = n, m
        else:
            current_matrix = 'Y'
            i, j = n, m
    else:  # local mode
        best_score = np.max(M)
        if best_score < 0:
            return (0, "", "")
        # Find the cell with maximum value
        max_i, max_j = np.unravel_index(np.argmax(M), M.shape)
        i, j = max_i, max_j
        current_matrix = 'M'
    aligned_a = []
    aligned_b = []
    while i > 0 or j > 0:
        if current_matrix == 'M':
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
            current_matrix = back_M[i][j]
        elif current_matrix == 'X':
            aligned_a.append(seq_a[i-1])
            aligned_b.append('-')
            i -= 1
            current_matrix = back_X[i][j]
        elif current_matrix == 'Y':
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            j -= 1
            current_matrix = back_Y[i][j]
    aligned_a = ''.join(reversed(aligned_a))
    aligned_b = ''.join(reversed(aligned_b))
    return (best_score, aligned_a, aligned_b)