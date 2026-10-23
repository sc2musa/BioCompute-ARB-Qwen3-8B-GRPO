def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    m = len(seq_a)
    n = len(seq_b)

    if not seq_a and not seq_b:
        return (0, "", "")
    elif not seq_a:
        score = gap_open + (n - 1) * gap_extend
        if mode == "local" and score <= 0:
            return (0, "", "")
        else:
            return (score, "-" * n, seq_b)
    elif not seq_b:
        score = gap_open + (m - 1) * gap_extend
        if mode == "local" and score <= 0:
            return (0, "", "")
        else:
            return (score, seq_a, "-" * m)

    if mode == "global":
        S = [[-float('inf')] * (n+1) for _ in range(m+1)]
        X = [[-float('inf')] * (n+1) for _ in range(m+1)]
        Y = [[-float('inf')] * (n+1) for _ in range(m+1)]

        S[0][0] = 0
        for i in range(1, m+1):
            X[i][0] = gap_open + (i-1)*gap_extend
            Y[i][0] = -float('inf')
        for j in range(1, n+1):
            Y[0][j] = gap_open + (j-1)*gap_extend
            X[0][j] = -float('inf')

        for i in range(1, m+1):
            for j in range(1, n+1):
                match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                S[i][j] = max(
                    S[i-1][j-1] + match_score,
                    X[i-1][j] + match_score,
                    Y[i][j-1] + match_score
                )
                X[i][j] = max(
                    X[i-1][j] + gap_extend,
                    S[i-1][j] + gap_open
                )
                Y[i][j] = max(
                    Y[i][j-1] + gap_extend,
                    S[i][j-1] + gap_open
                )

        i, j = m, n
        aligned_a = []
        aligned_b = []
        while i > 0 and j > 0:
            if S[i][j] == S[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif S[i][j] == X[i-1][j] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            elif S[i][j] == Y[i-1][j] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
            else:
                break

        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append('-')
            i -= 1
        while j > 0:
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            j -= 1

        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
        return (S[m][n], aligned_a, aligned_b)
    elif mode == "local":
        S = [[-float('inf')] * (n+1) for _ in range(m+1)]
        X = [[-float('inf')] * (n+1) for _ in range(m+1)]
        Y = [[-float('inf')] * (n+1) for _ in range(m+1)]
        S[0][0] = 0

        for i in range(1, m+1):
            X[i][0] = gap_open + (i-1)*gap_extend
            Y[i][0] = -float('inf')
        for j in range(1, n+1):
            Y[0][j] = gap_open + (j-1)*gap_extend
            X[0][j] = -float('inf')

        max_score = -float('inf')
        max_i, max_j = 0, 0
        for i in range(m+1):
            for j in range(n+1):
                current = max(S[i][j], X[i][j], Y[i][j])
                if current > max_score:
                    max_score = current
                    max_i, max_j = i, j

        i, j = max_i, max_j
        aligned_a = []
        aligned_b = []
        while i > 0 and j > 0:
            current_max = max(S[i][j], X[i][j], Y[i][j])
            if current_max == S[i][j]:
                match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif current_max == X[i][j]:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            elif current_max == Y[i][j]:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
            else:
                break

        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append('-')
            i -= 1
        while j > 0:
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            j -= 1

        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
        if max_score <= 0:
            return (0, "", "")
        else:
            return (max_score, aligned_a, aligned_b)
    else:
        raise ValueError("Invalid mode")

