def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)
    if len_a == 0 and len_b == 0:
        return (0, "", "")

    # Initialize DP tables
    if mode == "global":
        F = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        X = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        Y = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        F[0][0] = 0
        for i in range(1, len_a + 1):
            X[i][0] = F[i-1][0] + gap_open
        for j in range(1, len_b + 1):
            Y[0][j] = F[0][j-1] + gap_open
    else:  # local
        F = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        X = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        Y = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        F[0][0] = 0

    # Track sources for traceback
    sourceF = [[-1] * (len_b + 1) for _ in range(len_a + 1)]
    sourceX = [[-1] * (len_b + 1) for _ in range(len_a + 1)]
    sourceY = [[-1] * (len_b + 1) for _ in range(len_a + 1)]
    # 0: F, 1: X, 2: Y

    # Fill DP tables
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Compute score
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            # Update F[i][j]
            option1 = F[i-1][j-1] + score
            option2 = X[i-1][j] + score
            option3 = Y[i-1][j] + score
            F[i][j] = max(option1, option2, option3)
            if option1 >= option2 and option1 >= option3:
                sourceF[i][j] = 0
            elif option2 >= option1 and option2 >= option3:
                sourceF[i][j] = 1
            else:
                sourceF[i][j] = 2
            # Update X[i][j]
            optionX1 = X[i-1][j] + gap_extend
            optionX2 = F[i-1][j] + gap_open
            X[i][j] = max(optionX1, optionX2)
            if optionX1 >= optionX2:
                sourceX[i][j] = 1
            else:
                sourceX[i][j] = 0
            # Update Y[i][j]
            optionY1 = Y[i][j-1] + gap_extend
            optionY2 = F[i][j-1] + gap_open
            Y[i][j] = max(optionY1, optionY2)
            if optionY1 >= optionY2:
                sourceY[i][j] = 2
            else:
                sourceY[i][j] = 0
            # Local mode: allow starting new alignment
            if mode == "local":
                F[i][j] = max(F[i][j], 0)

    # Determine best score
    if mode == "global":
        best_score = max(F[len_a][len_b], X[len_a][len_b], Y[len_a][len_b])
        best_i, best_j = len_a, len_b
        if best_score < 0:
            return (0, "", "")
    else:  # local
        best_score = max(max(row) for row in F)
        if best_score <= 0:
            return (0, "", "")
        # Find the cell with the best score
        best_i, best_j = 0, 0
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if F[i][j] == best_score:
                    best_i, best_j = i, j
                    break
            else:
                continue
            break

    # Traceback
    aligned_a, aligned_b = [], []
    i, j = best_i, best_j
    while i > 0 and j > 0:
        if mode == "global":
            if F[i][j] >= X[i][j] and F[i][j] >= Y[i][j]:
                # From F
                if sourceF[i][j] == 0:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                elif sourceF[i][j] == 1:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    i -= 1
                else:
                    aligned_a.append("-")
                    aligned_b.append(seq_b[j-1])
                    j -= 1
            elif X[i][j] >= F[i][j] and X[i][j] >= Y[i][j]:
                # From X
                if sourceX[i][j] == 1:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    i -= 1
                else:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
            else:
                # From Y
                if sourceY[i][j] == 2:
                    aligned_a.append("-")
                    aligned_b.append(seq_b[j-1])
                    j -= 1
                else:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
        else:  # local
            if F[i][j] >= X[i][j] and F[i][j] >= Y[i][j]:
                # From F
                if sourceF[i][j] == 0:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                elif sourceF[i][j] == 1:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    i -= 1
                else:
                    aligned_a.append("-")
                    aligned_b.append(seq_b[j-1])
                    j -= 1
            elif X[i][j] >= F[i][j] and X[i][j] >= Y[i][j]:
                # From X
                if sourceX[i][j] == 1:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    i -= 1
                else:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
            else:
                # From Y
                if sourceY[i][j] == 2:
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
    aligned_a = ''.join(reversed(aligned_a))
    aligned_b = ''.join(reversed(aligned_b))
    return (best_score, aligned_a, aligned_b)