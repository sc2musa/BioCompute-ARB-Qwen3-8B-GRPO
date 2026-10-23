def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)
    INF = float('-inf')

    if mode == "global":
        # Global mode: Needleman-Wunsch
        M = [[INF]*(len_b+1) for _ in range(len_a+1)]
        X = [[INF]*(len_b+1) for _ in range(len_a+1)]
        Y = [[INF]*(len_b+1) for _ in range(len_a+1)]
        M[0][0] = 0
        # Initialize first column (Y) and first row (X)
        for i in range(1, len_a+1):
            X[i][0] = gap_open + (i-1)*gap_extend
        for j in range(1, len_b+1):
            Y[0][j] = gap_open + (j-1)*gap_extend
        for i in range(1, len_a+1):
            for j in range(1, len_b+1):
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                M[i][j] = max(M[i-1][j-1] + score, X[i-1][j] + match, Y[i][j-1] + match)
                X[i][j] = max(X[i-1][j] + gap_extend, M[i-1][j] + gap_open)
                Y[i][j] = max(Y[i][j-1] + gap_extend, M[i][j-1] + gap_open)
        score = M[len_a][len_b]
        # Backtrack
        i, j = len_a, len_b
        aligned_a, aligned_b = [], []
        while i > 0 and j > 0:
            if M[i][j] == M[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif M[i][j] == X[i-1][j] + match:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            elif M[i][j] == Y[i][j-1] + match:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append('-')
            i -= 1
        while j > 0:
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            j -= 1
        return (score, ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))
    elif mode == "local":
        # Local mode: Smith-Waterman
        S = [[INF]*(len_b+1) for _ in range(len_a+1)]
        M = [[INF]*(len_b+1) for _ in range(len_a+1)]
        X = [[INF]*(len_b+1) for _ in range(len_a+1)]
        Y = [[INF]*(len_b+1) for _ in range(len_a+1)]
        S[0][0] = 0
        max_score = 0
        max_i, max_j = 0, 0
        for i in range(len_a+1):
            for j in range(len_b+1):
                if i == 0 and j == 0:
                    continue
                # Compute M[i][j], X[i][j], Y[i][j] first
                if i > 0 and j > 0:
                    score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                    M[i][j] = max(M[i-1][j-1] + score, X[i-1][j] + match, Y[i][j-1] + match)
                    X[i][j] = max(X[i-1][j] + gap_extend, M[i-1][j] + gap_open)
                    Y[i][j] = max(Y[i][j-1] + gap_extend, M[i][j-1] + gap_open)
                elif i > 0:
                    X[i][j] = max(X[i-1][j] + gap_extend, M[i-1][j] + gap_open)
                    Y[i][j] = INF
                elif j > 0:
                    Y[i][j] = max(Y[i][j-1] + gap_extend, M[i][j-1] + gap_open)
                    X[i][j] = INF
                else:
                    X[i][j] = INF
                    Y[i][j] = INF
                # Now compute S[i][j]
                if i == 0 or j == 0:
                    score = 0
                else:
                    score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                S[i][j] = max(
                    M[i][j],
                    S[i-1][j] + gap_extend,
                    S[i][j-1] + gap_extend,
                    S[i-1][j-1] + score
                )
                if S[i][j] > max_score:
                    max_score = S[i][j]
                    max_i, max_j = i, j
        if max_score <= 0:
            return (0, "", "")
        # Backtrack
        i, j = max_i, max_j
        aligned_a, aligned_b = [], []
        while i > 0 and j > 0:
            if S[i][j] == S[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else 0):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif S[i][j] == S[i-1][j] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            elif S[i][j] == S[i][j-1] + gap_extend:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
            elif S[i][j] == M[i][j]:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            else:
                break
        return (max_score, ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))
    else:
        raise ValueError("Invalid mode")