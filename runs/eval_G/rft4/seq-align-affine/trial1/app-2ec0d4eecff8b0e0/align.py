def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    m, n = len(seq_a), len(seq_b)
    S = [[-float('inf')] * (n+1) for _ in range(m+1)]
    X = [[-float('inf')] * (n+1) for _ in range(m+1)]
    Y = [[-float('inf')] * (n+1) for _ in range(m+1)]

    S[0][0] = 0

    for i in range(m+1):
        for j in range(n+1):
            if i == 0 and j == 0:
                continue

            # Compute S[i][j]
            if i > 0 and j > 0:
                score_val = match if seq_a[i-1] == seq_b[j-1] else mismatch
                S[i][j] = max(S[i-1][j-1] + score_val, X[i-1][j-1] + score_val, Y[i-1][j-1] + score_val)
            
            # Compute X[i][j]
            if i == 0 and j > 0:
                X[i][j] = gap_open + (j-1)*gap_extend
            else:
                term1 = X[i-1][j] + gap_extend if i > 0 else -float('inf')
                term2 = Y[i][j-1] + gap_open if j > 0 else -float('inf')
                term3 = S[i][j-1] + gap_open if j > 0 else -float('inf')
                X[i][j] = max(term1, term2, term3)

            # Compute Y[i][j]
            if j == 0 and i > 0:
                Y[i][j] = gap_open + (i-1)*gap_extend
            else:
                term1 = Y[i][j-1] + gap_extend if j > 0 else -float('inf')
                term2 = X[i-1][j] + gap_open if i > 0 else -float('inf')
                term3 = S[i-1][j] + gap_open if i > 0 else -float('inf')
                Y[i][j] = max(term1, term2, term3)

    if mode == "global":
        max_score = max(S[m][n], X[m][n], Y[m][n])
        i, j = m, n
        if S[m][n] >= X[m][n] and S[m][n] >= Y[m][n]:
            current_matrix = "S"
        elif X[m][n] >= S[m][n] and X[m][n] >= Y[m][n]:
            current_matrix = "X"
        else:
            current_matrix = "Y"
    else:
        max_score = -float('inf')
        for i in range(m+1):
            for j in range(n+1):
                current = max(S[i][j], X[i][j], Y[i][j])
                if current > max_score:
                    max_score = current
                    max_i, max_j = i, j
        if max_score <= 0:
            # Check if there's a non-empty alignment with score 0
            if max_i > 0 or max_j > 0:
                i, j = max_i, max_j
                current_matrix = "S" if S[i][j] == max_score else "X" if X[i][j] == max_score else "Y"
            else:
                return (0, "", "")
        else:
            i, j = max_i, max_j
            current_matrix = "S" if S[i][j] == max_score else "X" if X[i][j] == max_score else "Y"

    # Backtrack
    aligned_a, aligned_b = [], []
    while i > 0 or j > 0:
        if current_matrix == "S":
            if i > 0 and j > 0:
                score_val = match if seq_a[i-1] == seq_b[j-1] else mismatch
                term1 = S[i-1][j-1] + score_val
                term2 = X[i-1][j-1] + score_val
                term3 = Y[i-1][j-1] + score_val
                if S[i][j] == term1:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                    current_matrix = "S"
                elif S[i][j] == term2:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                    current_matrix = "X"
                elif S[i][j] == term3:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                    current_matrix = "Y"
                else:
                    break
            else:
                break
        elif current_matrix == "X":
            if i > 0:
                term1 = X[i-1][j] + gap_extend
                term2 = Y[i][j-1] + gap_open
                term3 = S[i][j-1] + gap_open
                if X[i][j] == term1:
                    aligned_a.append("-")
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    current_matrix = "X"
                elif X[i][j] == term2:
                    aligned_a.append("-")
                    aligned_b.append(seq_b[j-1])
                    j -= 1
                    current_matrix = "Y"
                elif X[i][j] == term3:
                    aligned_a.append("-")
                    aligned_b.append(seq_b[j-1])
                    j -= 1
                    current_matrix = "S"
                else:
                    break
            else:
                break
        else:  # Y
            if j > 0:
                term1 = Y[i][j-1] + gap_extend
                term2 = X[i-1][j] + gap_open
                term3 = S[i-1][j] + gap_open
                if Y[i][j] == term1:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    j -= 1
                    current_matrix = "Y"
                elif Y[i][j] == term2:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    i -= 1
                    current_matrix = "X"
                elif Y[i][j] == term3:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    i -= 1
                    current_matrix = "S"
                else:
                    break
            else:
                break

    aligned_a = ''.join(reversed(aligned_a))
    aligned_b = ''.join(reversed(aligned_b))
    return (max_score, aligned_a, aligned_b)