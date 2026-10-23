import math

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    if not seq_a and not seq_b:
        return (0, "", "")
    elif not seq_a:
        score = gap_open + (len(seq_b) - 1) * gap_extend
        return (score, "-" * len(seq_b), seq_b)
    elif not seq_b:
        score = gap_open + (len(seq_a) - 1) * gap_extend
        return (score, seq_a, "-" * len(seq_a))

    m, n = len(seq_a), len(seq_b)
    INF = -math.inf

    if mode == "global":
        dp = [[INF] * (n+1) for _ in range(m+1)]
        gap_a = [[INF] * (n+1) for _ in range(m+1)]
        gap_b = [[INF] * (n+1) for _ in range(m+1)]

        dp[0][0] = 0
        for j in range(1, n+1):
            gap_a[0][j] = gap_open + (j-1)*gap_extend
        for i in range(1, m+1):
            gap_b[i][0] = gap_open + (i-1)*gap_extend

        for i in range(1, m+1):
            for j in range(1, n+1):
                if seq_a[i-1] == seq_b[j-1]:
                    score = match
                else:
                    score = mismatch
                dp[i][j] = max(dp[i-1][j-1] + score, gap_a[i-1][j] + score, gap_b[i-1][j] + score)

                gap_a[i][j] = max(gap_a[i-1][j] + gap_extend, gap_b[i-1][j] + gap_open + gap_extend, dp[i-1][j] + gap_open + gap_extend)

                gap_b[i][j] = max(gap_b[i][j-1] + gap_extend, gap_a[i][j-1] + gap_open + gap_extend, dp[i][j-1] + gap_open + gap_extend)

        final_score = max(dp[m][n], gap_a[m][n], gap_b[m][n])

        aligned_a, aligned_b = [], []
        i, j = m, n
        if final_score == dp[m][n]:
            while i > 0 and j > 0:
                if seq_a[i-1] == seq_b[j-1]:
                    score = match
                else:
                    score = mismatch
                if dp[i][j] == dp[i-1][j-1] + score:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                elif dp[i][j] == gap_a[i-1][j] + score:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                elif dp[i][j] == gap_b[i-1][j] + score:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    j -= 1
        elif final_score == gap_a[m][n]:
            while i > 0 and j > 0:
                if seq_a[i-1] == seq_b[j-1]:
                    score = match
                else:
                    score = mismatch
                if gap_a[i][j] == gap_a[i-1][j] + gap_extend:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                elif gap_a[i][j] == gap_b[i-1][j] + gap_open + gap_extend:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                elif gap_a[i][j] == dp[i-1][j] + gap_open + gap_extend:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
        else:
            while i > 0 and j > 0:
                if seq_a[i-1] == seq_b[j-1]:
                    score = match
                else:
                    score = mismatch
                if gap_b[i][j] == gap_b[i][j-1] + gap_extend:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    j -= 1
                elif gap_b[i][j] == gap_a[i][j-1] + gap_open + gap_extend:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    j -= 1
                elif gap_b[i][j] == dp[i][j-1] + gap_open + gap_extend:
                    aligned_a.append(seq_a[i-1])
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
        return (final_score, ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))
    else:  # local mode
        dp = [[INF] * (n+1) for _ in range(m+1)]
        gap_a = [[INF] * (n+1) for _ in range(m+1)]
        gap_b = [[INF] * (n+1) for _ in range(m+1)]
        dp[0][0] = 0
        for j in range(1, n+1):
            gap_a[0][j] = gap_open + (j-1)*gap_extend
        for i in range(1, m+1):
            gap_b[i][0] = gap_open + (i-1)*gap_extend
        for i in range(1, m+1):
            for j in range(1, n+1):
                if seq_a[i-1] == seq_b[j-1]:
                    score = match
                else:
                    score = mismatch
                dp[i][j] = max(dp[i-1][j-1] + score, gap_a[i-1][j] + score, gap_b[i-1][j] + score, 0)
                gap_a[i][j] = max(gap_a[i-1][j] + gap_extend, gap_b[i-1][j] + gap_open + gap_extend, dp[i-1][j] + gap_open + gap_extend)
                gap_b[i][j] = max(gap_b[i][j-1] + gap_extend, gap_a[i][j-1] + gap_open + gap_extend, dp[i][j-1] + gap_open + gap_extend)
        max_score = max(max(row) for row in dp)
        if max_score <= 0:
            return (0, "", "")
        # Backtrack
        i, j = 0, 0
        for i in range(m+1):
            for j in range(n+1):
                if dp[i][j] == max_score:
                    break
            else:
                continue
            break
        aligned_a, aligned_b = [], []
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            if dp[i][j] == dp[i-1][j-1] + score:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif dp[i][j] == gap_a[i-1][j] + score:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif dp[i][j] == gap_b[i-1][j] + score:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                j -= 1
            elif dp[i][j] == 0:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            else:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append('-')
            i -= 1
        while j > 0:
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            j -= 1
        return (max_score, ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))

