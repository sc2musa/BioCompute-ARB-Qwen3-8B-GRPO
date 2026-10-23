import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    if not seq_a and not seq_b:
        return (0, "", "")
    elif not seq_a:
        return (0, "", seq_b)
    elif not seq_b:
        return (0, seq_a, "")

    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.full((len_a + 1, len_b + 1, 3), -np.inf)
    dp[0][0][0] = 0

    # Initialize boundaries
    for j in range(1, len_b + 1):
        dp[0][j][1] = gap_open + (j - 1) * gap_extend
    for i in range(1, len_a + 1):
        dp[i][0][2] = gap_open + (i - 1) * gap_extend

    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Match/mismatch score
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # State 0: no gap
            dp[i][j][0] = max(
                dp[i-1][j-1][0] + score,
                dp[i-1][j][1] + score,
                dp[i][j-1][2] + score
            )
            
            # State 1: gap in seq_a
            dp[i][j][1] = max(
                dp[i-1][j-1][0] + gap_open,  # new gap in seq_a from state 0
                dp[i-1][j][1] + gap_extend,   # extend gap in seq_a from state 1
                dp[i-1][j][2] + gap_open      # new gap in seq_a from state 2
            )
            
            # State 2: gap in seq_b
            dp[i][j][2] = max(
                dp[i-1][j-1][0] + gap_open,  # new gap in seq_b from state 0
                dp[i][j-1][2] + gap_extend,   # extend gap in seq_b from state 2
                dp[i-1][j][1] + gap_open      # new gap in seq_b from state 1
            )

    # Determine final score
    if mode == "global":
        final_score = dp[len_a][len_b][0]
    else:
        max_score = np.max(dp)
        final_score = max_score if max_score > 0 else 0

    # Backtrack to find alignment
    aligned_a, aligned_b = "", ""
    if mode == "global":
        i, j, state = len_a, len_b, 0
    else:
        # Find max score position
        max_val = -np.inf
        max_i, max_j, max_state = 0, 0, 0
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                for s in range(3):
                    if dp[i][j][s] > max_val:
                        max_val = dp[i][j][s]
                        max_i, max_j, max_state = i, j, s
        i, j, state = max_i, max_j, max_state

    # Backtrack
    while i > 0 or j > 0:
        if state == 0:
            # Match/mismatch
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
            state = 0
        elif state == 1:
            # Gap in seq_a
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            state = 1
        else:
            # Gap in seq_b
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            j -= 1
            state = 2

    return (final_score, aligned_a, aligned_b)