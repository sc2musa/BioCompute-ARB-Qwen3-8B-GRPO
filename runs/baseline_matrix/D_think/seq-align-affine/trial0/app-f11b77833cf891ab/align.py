import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a = len(seq_a)
    len_b = len(seq_b)

    # Initialize DP table
    dp = np.full((len_a + 1, len_b + 1, 3), -np.inf, dtype=np.int64)
    dp[0][0][0] = 0

    # Initialize first row and column
    for j in range(1, len_b + 1):
        dp[0][j][1] = gap_open + (j - 1) * gap_extend
    for i in range(1, len_a + 1):
        dp[i][0][2] = gap_open + (i - 1) * gap_extend

    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # State 0: match/mismatch
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            dp[i][j][0] = max(
                dp[i-1][j-1][0] + score,
                dp[i-1][j][1] + score,
                dp[i][j-1][2] + score
            )
            
            # State 1: gap in a
            dp[i][j][1] = max(
                dp[i-1][j][1] + gap_extend,
                dp[i-1][j][0] + gap_open
            )
            
            # State 2: gap in b
            dp[i][j][2] = max(
                dp[i][j-1][2] + gap_extend,
                dp[i][j-1][0] + gap_open
            )

    # Determine maximum score based on mode
    if mode == "global":
        max_score = max(dp[len_a][len_b][0], dp[len_a][len_b][1], dp[len_a][len_b][2])
    else:  # local
        max_score = np.max(dp)
        if max_score < 0:
            return (0, "", "")

    # Backtrack to find alignment
    aligned_a = []
    aligned_b = []
    i, j, state = len_a, len_b, 0
    if mode == "global":
        if dp[len_a][len_b][0] >= dp[len_a][len_b][1] and dp[len_a][len_b][0] >= dp[len_a][len_b][2]:
            state = 0
        elif dp[len_a][len_b][1] >= dp[len_a][len_b][2]:
            state = 1
        else:
            state = 2
    else:
        # Find the position of the maximum score
        max_i, max_j, max_state = np.unravel_index(np.argmax(dp), dp.shape)
        i, j, state = max_i, max_j, max_state

    while i > 0 or j > 0:
        if state == 0:
            # Match/mismatch
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
            # Determine previous state
            if dp[i][j][0] == dp[i-1][j-1][0] + score:
                prev_state = 0
            elif dp[i][j][0] == dp[i-1][j][1] + score:
                prev_state = 1
            else:
                prev_state = 2
            state = prev_state
        elif state == 1:
            # Gap in a
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            i -= 1
            # Determine previous state
            if dp[i][j][1] == dp[i-1][j][1] + gap_extend:
                prev_state = 1
            else:
                prev_state = 0
            state = prev_state
        else:  # state == 2
            # Gap in b
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            j -= 1
            # Determine previous state
            if dp[i][j][2] == dp[i][j-1][2] + gap_extend:
                prev_state = 2
            else:
                prev_state = 0
            state = prev_state

    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]

    return (max_score, ''.join(aligned_a), ''.join(aligned_b))