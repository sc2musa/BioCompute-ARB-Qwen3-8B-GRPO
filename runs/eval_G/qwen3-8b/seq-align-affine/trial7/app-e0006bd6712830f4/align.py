def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)
    
    # Initialize DP tables
    match_dp = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
    gapA_dp = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
    gapB_dp = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
    match_dp[0][0] = 0
    
    # Fill DP tables
    for i in range(len_a + 1):
        for j in range(len_b + 1):
            if i == 0 and j == 0:
                continue
            if i == 0:
                gapA_dp[i][j] = gap_open if j == 1 else gapA_dp[i][j-1] + gap_extend
                gapB_dp[i][j] = -float('inf')
            elif j == 0:
                gapB_dp[i][j] = gap_open if i == 1 else gapB_dp[i-1][j] + gap_extend
                gapA_dp[i][j] = -float('inf')
            else:
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                match_dp[i][j] = max(
                    match_dp[i-1][j-1] + score,
                    gapA_dp[i-1][j] + score,
                    gapB_dp[i][j-1] + score
                )
                gapA_dp[i][j] = max(
                    gapA_dp[i-1][j] + gap_extend,
                    gapB_dp[i-1][j] + gap_open,
                    match_dp[i-1][j] + gap_open
                )
                gapB_dp[i][j] = max(
                    gapB_dp[i][j-1] + gap_extend,
                    gapA_dp[i][j-1] + gap_open,
                    match_dp[i][j-1] + gap_open
                )
    
    # Determine score and alignment
    if mode == "global":
        score = match_dp[len_a][len_b]
        aligned_a, aligned_b = backtrack_global(len_a, len_b, seq_a, seq_b, match_dp, gapA_dp, gapB_dp, match, mismatch, gap_open, gap_extend)
    else:
        max_score = -float('inf')
        max_i, max_j = 0, 0
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if match_dp[i][j] > max_score:
                    max_score = match_dp[i][j]
                    max_i, max_j = i, j
                if gapA_dp[i][j] > max_score:
                    max_score = gapA_dp[i][j]
                    max_i, max_j = i, j
                if gapB_dp[i][j] > max_score:
                    max_score = gapB_dp[i][j]
                    max_i, max_j = i, j
        if max_score <= 0:
            return (0, "", "")
        aligned_a, aligned_b = backtrack_local(max_i, max_j, seq_a, seq_b, match_dp, gapA_dp, gapB_dp, match, mismatch, gap_open, gap_extend)
        score = max_score
    
    return (score, aligned_a, aligned_b)


def backtrack_global(len_a, len_b, seq_a, seq_b, match_dp, gapA_dp, gapB_dp, match, mismatch, gap_open, gap_extend):
    i, j = len_a, len_b
    aligned_a, aligned_b = [], []
    while i > 0 and j > 0:
        score = match if seq_a[i-1] == seq_b[j-1] else mismatch
        if match_dp[i][j] == match_dp[i-1][j-1] + score:
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif match_dp[i][j] == gapA_dp[i-1][j] + score:
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            i -= 1
        elif match_dp[i][j] == gapB_dp[i][j-1] + score:
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
    return ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b))


def backtrack_local(max_i, max_j, seq_a, seq_b, match_dp, gapA_dp, gapB_dp, match, mismatch, gap_open, gap_extend):
    i, j = max_i, max_j
    aligned_a, aligned_b = [], []
    while i > 0 and j > 0:
        score = match if seq_a[i-1] == seq_b[j-1] else mismatch
        if match_dp[i][j] == match_dp[i-1][j-1] + score:
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif match_dp[i][j] == gapA_dp[i-1][j] + score:
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            i -= 1
        elif match_dp[i][j] == gapB_dp[i][j-1] + score:
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
    return ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b))