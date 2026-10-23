def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    import sys
    INF = -sys.maxsize
    len_a, len_b = len(seq_a), len(seq_b)

    def score(x, y):
        return match if x == y else mismatch

    # Initialize DP tables
    dp = [[[INF for _ in range(3)] for _ in range(len_b + 1)] for _ in range(len_a + 1)]
    dp[0][0][0] = 0

    for j in range(1, len_b + 1):
        dp[0][j][1] = gap_open + (j - 1) * gap_extend
    for i in range(1, len_a + 1):
        dp[i][0][2] = gap_open + (i - 1) * gap_extend

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # State 0: match/mismatch
            current_score = score(seq_a[i-1], seq_b[j-1])
            dp[i][j][0] = max(
                dp[i-1][j-1][0] + current_score,
                dp[i-1][j][2] + current_score,
                dp[i][j-1][1] + current_score
            )
            # State 1: gap in A
            dp[i][j][1] = max(
                dp[i-1][j][1] + gap_extend,
                dp[i-1][j-1][0] + gap_open
            )
            # State 2: gap in B
            dp[i][j][2] = max(
                dp[i][j-1][2] + gap_extend,
                dp[i-1][j-1][0] + gap_open
            )

    if mode == "global":
        max_score = max(dp[len_a][len_b])
        i, j = len_a, len_b
        if dp[len_a][len_b][0] == max_score:
            state = 0
        elif dp[len_a][len_b][1] == max_score:
            state = 1
        else:
            state = 2
    else:
        max_score = -INF
        best_i, best_j, best_state = 0, 0, 0
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                for s in range(3):
                    if dp[i][j][s] > max_score:
                        max_score = dp[i][j][s]
                        best_i, best_j, best_state = i, j, s
        if max_score <= 0:
            return (0, "", "")
        i, j, state = best_i, best_j, best_state

    # Backtrack
    aligned_a, aligned_b = [], []
    while i > 0 or j > 0:
        if state == 0:
            # Check which option led to this state
            if dp[i][j][0] == dp[i-1][j-1][0] + score(seq_a[i-1], seq_b[j-1]):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
                state = 0
            elif dp[i][j][0] == dp[i-1][j][2] + score(seq_a[i-1], seq_b[j-1]):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j = j
                state = 2
            else:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i = i
                j -= 1
                state = 1
        elif state == 1:
            # Gap in A
            if dp[i][j][1] == dp[i-1][j][1] + gap_extend:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
                j = j
                state = 1
            else:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
                state = 0
        else:
            # Gap in B
            if dp[i][j][2] == dp[i][j-1][2] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i = i
                j -= 1
                state = 2
            else:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
                j -= 1
                state = 0
    return (max_score, ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))