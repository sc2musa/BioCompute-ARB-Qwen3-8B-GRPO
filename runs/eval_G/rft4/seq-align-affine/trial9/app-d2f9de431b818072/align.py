def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)
    if not seq_a and not seq_b:
        return (0, "", "")
    elif not seq_a:
        if mode == "global":
            score = gap_open + (len_b - 1) * gap_extend
            return (score, "", seq_b)
        else:
            return (0, "", "")
    elif not seq_b:
        if mode == "global":
            score = gap_open + (len_a - 1) * gap_extend
            return (score, seq_a, "")
        else:
            return (0, "", "")

    def backtrack(M, X, Y, i, j, match, mismatch, gap_open, gap_extend):
        aligned_a = []
        aligned_b = []
        while i > 0 and j > 0:
            # Check M[i][j]
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            option1 = M[i-1][j-1] + match_score
            option2 = X[i-1][j-1] + match_score
            option3 = Y[i-1][j-1] + match_score
            if M[i][j] == option1:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif M[i][j] == option2:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            elif M[i][j] == option3:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
            # Check X[i][j]
            option1 = X[i-1][j] + gap_extend
            option2 = Y[i-1][j] + gap_open + gap_extend
            option3 = M[i-1][j] + gap_open + gap_extend
            if X[i][j] == option1:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            elif X[i][j] == option2:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            elif X[i][j] == option3:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            # Check Y[i][j]
            option1 = Y[i][j-1] + gap_extend
            option2 = X[i][j-1] + gap_open + gap_extend
            option3 = M[i][j-1] + gap_open + gap_extend
            if Y[i][j] == option1:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
            elif Y[i][j] == option2:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
            elif Y[i][j] == option3:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
        # Handle remaining characters
        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append('-')
            i -= 1
        while j > 0:
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            j -= 1
        return (aligned_a, aligned_b)

    if mode == "local":
        # Local mode: Smith-Waterman
        M = [[0]*(len_b+1) for _ in range(len_a+1)]
        X = [[0]*(len_b+1) for _ in range(len_a+1)]
        Y = [[0]*(len_b+1) for _ in range(len_a+1)]
        for i in range(1, len_a+1):
            for j in range(1, len_b+1):
                match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                M[i][j] = max(
                    M[i-1][j-1] + match_score,
                    X[i-1][j-1] + match_score,
                    Y[i-1][j-1] + match_score
                )
                X[i][j] = max(
                    X[i-1][j] + gap_extend,
                    Y[i-1][j] + gap_open + gap_extend,
                    M[i-1][j] + gap_open + gap_extend
                )
                Y[i][j] = max(
                    Y[i][j-1] + gap_extend,
                    X[i][j-1] + gap_open + gap_extend,
                    M[i][j-1] + gap_open + gap_extend
                )
        max_score = 0
        max_i, max_j = 0, 0
        for i in range(len_a+1):
            for j in range(len_b+1):
                current = max(M[i][j], X[i][j], Y[i][j])
                if current > max_score:
                    max_score = current
                    max_i, max_j = i, j
        if max_score == 0:
            return (0, "", "")
        aligned_a, aligned_b = backtrack(M, X, Y, max_i, max_j, match, mismatch, gap_open, gap_extend)
        return (max_score, ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))
    else:  # Global mode: Needleman-Wunsch
        M = [[0]*(len_b+1) for _ in range(len_a+1)]
        X = [[0]*(len_b+1) for _ in range(len_a+1)]
        Y = [[0]*(len_b+1) for _ in range(len_a+1)]
        for j in range(1, len_b+1):
            X[0][j] = gap_open + (j-1)*gap_extend
        for i in range(1, len_a+1):
            Y[i][0] = gap_open + (i-1)*gap_extend
        for i in range(1, len_a+1):
            for j in range(1, len_b+1):
                match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                M[i][j] = max(
                    M[i-1][j-1] + match_score,
                    X[i-1][j-1] + match_score,
                    Y[i-1][j-1] + match_score
                )
                X[i][j] = max(
                    X[i-1][j] + gap_extend,
                    Y[i-1][j] + gap_open + gap_extend,
                    M[i-1][j] + gap_open + gap_extend
                )
                Y[i][j] = max(
                    Y[i][j-1] + gap_extend,
                    X[i][j-1] + gap_open + gap_extend,
                    M[i][j-1] + gap_open + gap_extend
                )
        score = M[len_a][len_b]
        aligned_a, aligned_b = backtrack(M, X, Y, len_a, len_b, match, mismatch, gap_open, gap_extend)
        return (score, ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))
