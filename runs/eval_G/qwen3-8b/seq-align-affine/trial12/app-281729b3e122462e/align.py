def align(seq_a: str, seq_b: str, mode: str = "global",
              match: int = 2, mismatch: int = -3,
              gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)
    if mode == "global":
        dp = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        dp_gapA = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        dp_gapB = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        dp[0][0] = 0
        # Initialize first row (i=0)
        for j in range(1, len_b + 1):
            dp_gapA[0][j] = gap_open + (j-1)*gap_extend
            dp_gapB[0][j] = gap_open + (j-1)*gap_extend
        # Initialize first column (j=0)
        for i in range(1, len_a + 1):
            dp_gapA[i][0] = gap_open + (i-1)*gap_extend
            dp_gapB[i][0] = gap_open + (i-1)*gap_extend
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if i > 0 and j > 0:
                    score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                    dp[i][j] = max(dp[i-1][j-1] + score, dp_gapA[i-1][j] + score, dp_gapB[i-1][j] + score)
                if i > 0:
                    dp_gapA[i][j] = max(dp_gapA[i-1][j] + gap_extend, dp[i-1][j] + gap_open)
                if j > 0:
                    dp_gapB[i][j] = max(dp_gapB[i][j-1] + gap_extend, dp[i][j-1] + gap_open)
        score = dp[len_a][len_b]
        i, j = len_a, len_b
        aligned_a, aligned_b = [], []
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                option1 = dp[i-1][j-1] + score
                option2 = dp_gapA[i-1][j] + score
                option3 = dp_gapB[i-1][j] + score
                if dp[i][j] == option1:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                elif dp[i][j] == option2:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                else:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    j -= 1
            elif i > 0:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            else:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
        return (score, ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))
    else:
        dp = [[0]*(len_b + 1) for _ in range(len_a + 1)]
        dp_gapA = [[0]*(len_b + 1) for _ in range(len_a + 1)]
        dp_gapB = [[0]*(len_b + 1) for _ in range(len_a + 1)]
        max_score = 0
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if i > 0 and j > 0:
                    score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                    dp[i][j] = max(dp[i-1][j-1] + score, dp_gapA[i-1][j] + score, dp_gapB[i-1][j] + score)
                if i > 0:
                    dp_gapA[i][j] = max(dp_gapA[i-1][j] + gap_extend, dp[i-1][j] + gap_open)
                if j > 0:
                    dp_gapB[i][j] = max(dp_gapB[i][j-1] + gap_extend, dp[i][j-1] + gap_open)
                if dp[i][j] > max_score:
                    max_score = dp[i][j]
        if max_score == 0:
            return (0, "", "")
        i, j = 0, 0
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if dp[i][j] == max_score:
                    break
            else:
                continue
            break
        aligned_a, aligned_b = [], []
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                option1 = dp[i-1][j-1] + score
                option2 = dp_gapA[i-1][j] + score
                option3 = dp_gapB[i-1][j] + score
                if dp[i][j] == option1:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                elif dp[i][j] == option2:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                else:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    j -= 1
            elif i > 0:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            else:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
        return (max_score, ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))