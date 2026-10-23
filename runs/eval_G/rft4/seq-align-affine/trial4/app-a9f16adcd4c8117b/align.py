def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)
    if len_a == 0 and len_b == 0:
        return (0, "", "")

    if mode == "global":
        dp = [[0]*(len_b+1) for _ in range(len_a+1)]
        gapA = [[-float('inf')]*(len_b+1) for _ in range(len_a+1)]
        gapB = [[-float('inf')]*(len_b+1) for _ in range(len_a+1)]

        for j in range(1, len_b+1):
            gapB[0][j] = gap_open + (j-1)*gap_extend
        for i in range(1, len_a+1):
            gapA[i][0] = gap_open + (i-1)*gap_extend

        for i in range(1, len_a+1):
            for j in range(1, len_b+1):
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                dp[i][j] = max(
                    dp[i-1][j-1] + score,
                    gapA[i-1][j-1] + score,
                    gapB[i-1][j-1] + score
                )
                gapA[i][j] = max(
                    gapA[i-1][j] + gap_extend,
                    dp[i-1][j] + gap_open,
                    gapB[i-1][j] + gap_open
                )
                gapB[i][j] = max(
                    gapB[i][j-1] + gap_extend,
                    dp[i][j-1] + gap_open,
                    gapA[i][j-1] + gap_open
                )

        score = dp[len_a][len_b]
        aligned_a, aligned_b = [], []
        i, j = len_a, len_b
        while i > 0 and j > 0:
            if dp[i][j] == max(dp[i][j], gapA[i][j], gapB[i][j]):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif gapA[i][j] == max(dp[i][j], gapA[i][j], gapB[i][j]):
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
            else:
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
        return (score, ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))
    else:
        dp = [[0]*(len_b+1) for _ in range(len_a+1)]
        gapA = [[0]*(len_b+1) for _ in range(len_a+1)]
        gapB = [[0]*(len_b+1) for _ in range(len_a+1)]

        for i in range(1, len_a+1):
            for j in range(1, len_b+1):
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                dp[i][j] = max(
                    dp[i-1][j-1] + score,
                    gapA[i-1][j-1] + score,
                    gapB[i-1][j-1] + score,
                    0
                )
                gapA[i][j] = max(
                    gapA[i-1][j] + gap_extend,
                    dp[i-1][j] + gap_open,
                    gapB[i-1][j] + gap_open,
                    0
                )
                gapB[i][j] = max(
                    gapB[i][j-1] + gap_extend,
                    dp[i][j-1] + gap_open,
                    gapA[i][j-1] + gap_open,
                    0
                )

        max_score = -float('inf')
        for i in range(len_a+1):
            for j in range(len_b+1):
                current = max(dp[i][j], gapA[i][j], gapB[i][j])
                if current > max_score:
                    max_score = current
        if max_score < 0:
            return (0, "", "")

        # Find the cell with max_score across all tables
        max_i, max_j = 0, 0
        for i in range(len_a+1):
            for j in range(len_b+1):
                if dp[i][j] == max_score or gapA[i][j] == max_score or gapB[i][j] == max_score:
                    max_i, max_j = i, j
                    break
            else:
                continue
            break

        aligned_a, aligned_b = [], []
        i, j = max_i, max_j
        while i > 0 and j > 0:
            if dp[i][j] == max(dp[i][j], gapA[i][j], gapB[i][j]):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif gapA[i][j] == max(dp[i][j], gapA[i][j], gapB[i][j]):
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
            else:
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
        return (max_score, ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))

