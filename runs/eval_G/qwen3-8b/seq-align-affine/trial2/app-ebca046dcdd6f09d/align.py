def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """
    Return (score, aligned_a, aligned_b).
    """
    m = len(seq_a)
    n = len(seq_b)

    def get_score(a, b):
        return match if a == b else mismatch

    if mode == "global":
        import sys
        INF = -sys.maxsize
        match_dp = [[INF]*(n+1) for _ in range(m+1)]
        gapA_dp = [[INF]*(n+1) for _ in range(m+1)]
        gapB_dp = [[INF]*(n+1) for _ in range(m+1)]
        match_dp[0][0] = 0
        for j in range(1, n+1):
            gapA_dp[0][j] = gap_open + (j-1)*gap_extend
        for i in range(1, m+1):
            gapB_dp[i][0] = gap_open + (i-1)*gap_extend
        for i in range(1, m+1):
            for j in range(1, n+1):
                score = get_score(seq_a[i-1], seq_b[j-1])
                match_dp[i][j] = max(
                    match_dp[i-1][j-1] + score,
                    gapA_dp[i-1][j-1] + score,
                    gapB_dp[i-1][j-1] + score
                )
                gapA_dp[i][j] = max(
                    match_dp[i-1][j] + gap_open,
                    gapA_dp[i-1][j] + gap_extend,
                    gapB_dp[i-1][j] + gap_open
                )
                gapB_dp[i][j] = max(
                    match_dp[i][j-1] + gap_open,
                    gapB_dp[i][j-1] + gap_extend,
                    gapA_dp[i][j-1] + gap_open
                )
        aligned_a = []
        aligned_b = []
        i, j = m, n
        while i > 0 and j > 0:
            current_max = max(match_dp[i][j], gapA_dp[i][j], gapB_dp[i][j])
            if match_dp[i][j] == current_max:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif gapA_dp[i][j] == current_max:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif gapB_dp[i][j] == current_max:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                j -= 1
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
        score = match_dp[m][n]
    elif mode == "local":
        match_dp = [[0]*(n+1) for _ in range(m+1)]
        gapA_dp = [[0]*(n+1) for _ in range(m+1)]
        gapB_dp = [[0]*(n+1) for _ in range(m+1)]
        for j in range(1, n+1):
            gapA_dp[0][j] = gap_open + (j-1)*gap_extend
        for i in range(1, m+1):
            gapB_dp[i][0] = gap_open + (i-1)*gap_extend
        for i in range(1, m+1):
            for j in range(1, n+1):
                score = get_score(seq_a[i-1], seq_b[j-1])
                match_dp[i][j] = max(
                    match_dp[i-1][j-1] + score,
                    gapA_dp[i-1][j-1] + score,
                    gapB_dp[i-1][j-1] + score
                )
                gapA_dp[i][j] = max(
                    match_dp[i-1][j] + gap_open,
                    gapA_dp[i-1][j] + gap_extend,
                    gapB_dp[i-1][j] + gap_open
                )
                gapB_dp[i][j] = max(
                    match_dp[i][j-1] + gap_open,
                    gapB_dp[i][j-1] + gap_extend,
                    gapA_dp[i][j-1] + gap_open
                )
        max_score = 0
        max_i, max_j = 0, 0
        for i in range(m+1):
            for j in range(n+1):
                current_max = max(match_dp[i][j], gapA_dp[i][j], gapB_dp[i][j])
                if current_max > max_score:
                    max_score = current_max
                    max_i, max_j = i, j
        if max_score <= 0:
            return (0, "", "")
        aligned_a = []
        aligned_b = []
        i, j = max_i, max_j
        while i > 0 and j > 0:
            current_max = max(match_dp[i][j], gapA_dp[i][j], gapB_dp[i][j])
            if match_dp[i][j] == current_max:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif gapA_dp[i][j] == current_max:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif gapB_dp[i][j] == current_max:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                j -= 1
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
        score = max_score
    return (score, aligned_a, aligned_b)