import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    m = len(seq_a)
    n = len(seq_b)
    INF = float('-inf')

    # Initialize DP and prev arrays
    dp = [[[INF for _ in range(3)] for _ in range(n+1)] for _ in range(m+1)]
    prev = [[[None for _ in range(3)] for _ in range(n+1)] for _ in range(m+1)]

    dp[0][0][0] = 0
    prev[0][0][0] = None

    for i in range(m+1):
        for j in range(n+1):
            if i == 0 and j == 0:
                continue
            # Compute state 0 (match/mismatch)
            if i > 0 and j > 0:
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                candidates = []
                # From state 0 (i-1,j-1)
                if dp[i-1][j-1][0] != INF:
                    candidates.append((dp[i-1][j-1][0] + score, (i-1, j-1, 0)))
                # From state 1 (i-1,j)
                if dp[i-1][j][1] != INF:
                    candidates.append((dp[i-1][j][1] + score, (i-1, j, 1)))
                # From state 2 (i,j-1)
                if dp[i][j-1][2] != INF:
                    candidates.append((dp[i][j-1][2] + score, (i, j-1, 2)))
                if candidates:
                    max_val, prev_cell = max(candidates)
                    dp[i][j][0] = max_val
                    prev[i][j][0] = prev_cell
            # Compute state 1 (gap in seq_a)
            if i > 0:
                candidates = []
                # From state 1 (i-1,j)
                if dp[i-1][j][1] != INF:
                    candidates.append((dp[i-1][j][1] + gap_extend, (i-1, j, 1)))
                # From state 0 (i,j-1)
                if dp[i][j-1][0] != INF:
                    candidates.append((dp[i][j-1][0] + gap_open, (i, j-1, 0)))
                if candidates:
                    max_val, prev_cell = max(candidates)
                    dp[i][j][1] = max_val
                    prev[i][j][1] = prev_cell
            # Compute state 2 (gap in seq_b)
            if j > 0:
                candidates = []
                # From state 2 (i,j-1)
                if dp[i][j-1][2] != INF:
                    candidates.append((dp[i][j-1][2] + gap_extend, (i, j-1, 2)))
                # From state 0 (i-1,j)
                if dp[i-1][j][0] != INF:
                    candidates.append((dp[i-1][j][0] + gap_open, (i-1, j, 0)))
                if candidates:
                    max_val, prev_cell = max(candidates)
                    dp[i][j][2] = max_val
                    prev[i][j][2] = prev_cell

    # Determine the final score and backtracking start point
    if mode == "global":
        final_score = max(dp[m][n][0], dp[m][n][1], dp[m][n][2])
        start_i, start_j = m, n
        # Determine the initial state for backtracking
        max_val = max(dp[start_i][start_j][0], dp[start_i][start_j][1], dp[start_i][start_j][2])
        if max_val == dp[start_i][start_j][0]:
            current_state = 0
        elif max_val == dp[start_i][start_j][1]:
            current_state = 1
        else:
            current_state = 2
    else:  # local
        max_score = INF
        max_i, max_j = 0, 0
        for i in range(m+1):
            for j in range(n+1):
                for s in range(3):
                    if dp[i][j][s] > max_score:
                        max_score = dp[i][j][s]
                        max_i, max_j = i, j
        if max_score <= 0:
            return (0, "", "")
        start_i, start_j = max_i, max_j
        # Determine the initial state for backtracking
        max_val = dp[start_i][start_j][0]
        current_state = 0
        for s in [1, 2]:
            if dp[start_i][start_j][s] > max_val:
                max_val = dp[start_i][start_j][s]
                current_state = s

    # Backtrack to reconstruct the alignment
    aligned_a = []
    aligned_b = []
    current_i, current_j = start_i, start_j
    while current_i > 0 or current_j > 0:
        if current_state == 0:
            # Match/mismatch
            aligned_a.append(seq_a[current_i-1])
            aligned_b.append(seq_b[current_j-1])
            current_i -= 1
            current_j -= 1
            # Get the previous state
            prev_cell = prev[current_i][current_j][0]
            if prev_cell:
                current_i, current_j, current_state = prev_cell
        elif current_state == 1:
            # Gap in seq_a
            aligned_a.append('-')
            aligned_b.append(seq_b[current_j-1])
            current_i -= 1
            # Get the previous state
            prev_cell = prev[current_i][current_j][1]
            if prev_cell:
                current_i, current_j, current_state = prev_cell
        else:  # current_state == 2
            # Gap in seq_b
            aligned_a.append(seq_a[current_i-1])
            aligned_b.append('-')
            current_j -= 1
            # Get the previous state
            prev_cell = prev[current_i][current_j][2]
            if prev_cell:
                current_i, current_j, current_state = prev_cell

    # Reverse the aligned sequences
    aligned_a.reverse()
    aligned_b.reverse()

    # Return the score and aligned sequences
    if mode == "global":
        return (final_score, ''.join(aligned_a), ''.join(aligned_b))
    else:
        return (max_score, ''.join(aligned_a), ''.join(aligned_b))

# Run the check script
import subprocess
subprocess.run(["python", "check.py"])

# Submit the solution
submit()