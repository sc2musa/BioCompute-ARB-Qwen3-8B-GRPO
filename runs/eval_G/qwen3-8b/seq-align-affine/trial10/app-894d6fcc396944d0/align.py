def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    import numpy as np
    len_a, len_b = len(seq_a), len(seq_b)
    # Initialize DP table
    dp = np.full((len_a + 1, len_b + 1, 3), -np.inf)
    # Initialize based on mode
    if mode == "global":
        dp[0][0][0] = 0
        for j in range(1, len_b + 1):
            dp[0][j][1] = gap_open + (j - 1) * gap_extend
        for i in range(1, len_a + 1):
            dp[i][0][2] = gap_open + (i - 1) * gap_extend
    else:  # local mode
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                dp[i][j][0] = 0
                dp[i][j][1] = 0
                dp[i][j][2] = 0
    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Compute match score
            score_match = match if seq_a[i-1] == seq_b[j-1] else mismatch
            # State 0: match
            dp[i][j][0] = max(
                dp[i-1][j-1][0] + score_match,
                dp[i-1][j-1][1] + score_match,
                dp[i-1][j-1][2] + score_match
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
            # Local mode: ensure non-negative values
            if mode == "local":
                dp[i][j][0] = max(dp[i][j][0], 0)
                dp[i][j][1] = max(dp[i][j][1], 0)
                dp[i][j][2] = max(dp[i][j][2], 0)
    # Determine optimal score and backtrack
    if mode == "global":
        score = dp[len_a][len_b][0]
        i, j, state = len_a, len_b, 0
    else:
        max_score = np.max(dp)
        if max_score <= 0:
            return (0, "", "")
        # Find the cell with max score
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                for s in range(3):
                    if dp[i][j][s] == max_score:
                        i, j, state = i, j, s
                        break
                else:
                    continue
                break
        score = max_score
    # Backtrack to get alignment
    aligned_a = []
    aligned_b = []
    while i > 0 or j > 0:
        if state == 0:
            # Match
            if i > 0 and j > 0:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            else:
                break
            # Determine previous state
            current_val = dp[i][j][0]
            previous_val = current_val - score_match
            for s in [0, 1, 2]:
                if dp[i-1][j-1][s] == previous_val:
                    state = s
                    break
        elif state == 1:
            # Gap in A
            if j > 0:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                if dp[i-1][j][1] + gap_extend == dp[i][j][1]:
                    i -= 1
                    state = 1
                else:
                    i -= 1
                    j -= 1
                    state = 0
            else:
                break
        elif state == 2:
            # Gap in B
            if i > 0:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                if dp[i][j-1][2] + gap_extend == dp[i][j][2]:
                    j -= 1
                    state = 2
                else:
                    i -= 1
                    j -= 1
                    state = 0
            else:
                break
    aligned_a.reverse()
    aligned_b.reverse()
    return (score, ''.join(aligned_a), ''.join(aligned_b))