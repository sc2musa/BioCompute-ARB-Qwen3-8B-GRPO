def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    m, n = len(seq_a), len(seq_b)
    S = [[-float('inf')] * (n+1) for _ in range(m+1)]
    X = [[-float('inf')] * (n+1) for _ in range(m+1)]
    Y = [[-float('inf')] * (n+1) for _ in range(m+1)]

    if mode == "global":
        S[0][0] = 0
        for j in range(1, n+1):
            X[0][j] = -float('inf')
            Y[0][j] = -float('inf')
        for i in range(1, m+1):
            X[i][0] = -float('inf')
            Y[i][0] = -float('inf')
    else:
        S[0][0] = 0
        for j in range(1, n+1):
            X[0][j] = -float('inf')
            Y[0][j] = -float('inf')
        for i in range(1, m+1):
            X[i][0] = -float('inf')
            Y[i][0] = -float('inf')

    for i in range(1, m+1):
        for j in range(1, n+1):
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Compute S[i][j]
            S[i][j] = max(S[i-1][j-1] + score, X[i-1][j] + score, Y[i-1][j] + score)
            
            # Compute X[i][j]
            X[i][j] = max(X[i-1][j] + gap_extend, Y[i-1][j-1] + gap_open)
            
            # Compute Y[i][j]
            Y[i][j] = max(Y[i][j-1] + gap_extend, S[i-1][j-1] + gap_open)

    if mode == "global":
        best_score = S[m][n]
        i, j = m, n
    else:
        max_score = -float('inf')
        max_i, max_j = 0, 0
        for i in range(m+1):
            for j in range(n+1):
                current = max(S[i][j], X[i][j], Y[i][j])
                if current > max_score:
                    max_score = current
                    max_i, max_j = i, j
        if max_score <= 0:
            return (0, "", "")
        best_score = max_score
        i, j = max_i, max_j

    # Backtrack
    aligned_a, aligned_b = [], []
    while i > 0 or j > 0:
        current_max = max(S[i][j], X[i][j], Y[i][j])
        if current_max == S[i][j]:
            # Check which option contributed to S[i][j]
            if i > 0 and j > 0:
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                option1 = S[i-1][j-1] + score
                option2 = X[i-1][j] + score
                option3 = Y[i-1][j] + score
                if S[i][j] == option1:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                elif S[i][j] == option2:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    i -= 1
                elif S[i][j] == option3:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    j -= 1
            elif i > 0:
                option1 = X[i-1][j] + gap_extend
                option2 = Y[i-1][j-1] + gap_open
                if X[i][j] == option1:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    i -= 1
                elif X[i][j] == option2:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    j -= 1
            elif j > 0:
                option1 = Y[i][j-1] + gap_extend
                option2 = S[i-1][j-1] + gap_open
                if Y[i][j] == option1:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    j -= 1
                elif Y[i][j] == option2:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    i -= 1
        elif current_max == X[i][j]:
            # Check which option contributed to X[i][j]
            if i > 0 and j > 0:
                option1 = X[i-1][j] + gap_extend
                option2 = Y[i-1][j-1] + gap_open
                if X[i][j] == option1:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    i -= 1
                elif X[i][j] == option2:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    j -= 1
            elif i > 0:
                option1 = X[i-1][j] + gap_extend
                if X[i][j] == option1:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    i -= 1
            elif j > 0:
                option1 = Y[i-1][j-1] + gap_open
                if X[i][j] == option1:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    j -= 1
        elif current_max == Y[i][j]:
            # Check which option contributed to Y[i][j]
            if i > 0 and j > 0:
                option1 = Y[i][j-1] + gap_extend
                option2 = S[i-1][j-1] + gap_open
                if Y[i][j] == option1:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    j -= 1
                elif Y[i][j] == option2:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    i -= 1
            elif j > 0:
                option1 = Y[i][j-1] + gap_extend
                if Y[i][j] == option1:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    j -= 1
            elif i > 0:
                option1 = S[i-1][j-1] + gap_open
                if Y[i][j] == option1:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    i -= 1
        else:
            break
    
    return (best_score, ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))

