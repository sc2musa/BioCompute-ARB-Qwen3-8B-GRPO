import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)
    INF = float('-inf')

    # Initialize DP tables
    if mode == "global":
        state0 = [[0]*(len_b+1) for _ in range(len_a+1)]
        state1 = [[INF]*(len_b+1) for _ in range(len_a+1)]
        state2 = [[INF]*(len_b+1) for _ in range(len_a+1)]
    else:
        state0 = [[0]*(len_b+1) for _ in range(len_a+1)]
        state1 = [[0]*(len_b+1) for _ in range(len_a+1)]
        state2 = [[0]*(len_b+1) for _ in range(len_a+1)]

    # Fill DP tables
    for i in range(len_a + 1):
        for j in range(len_b + 1):
            if i == 0 and j == 0:
                continue

            # Compute state0[i][j]
            score_val = match if (seq_a[i-1] == seq_b[j-1] if i > 0 and j > 0 else 0) else mismatch
            val0 = INF
            if i > 0 and j > 0:
                val0 = state0[i-1][j-1] + score_val
            if i > 0:
                val0 = max(val0, state1[i-1][j] + score_val)
            if j > 0:
                val0 = max(val0, state2[i][j-1] + score_val)
            state0[i][j] = val0

            # Compute state1[i][j]
            val1 = INF
            if i > 0:
                val1 = state0[i-1][j] + gap_open
            if i > 0 and j > 0:
                val1 = max(val1, state1[i-1][j] + gap_extend)
            state1[i][j] = val1

            # Compute state2[i][j]
            val2 = INF
            if j > 0:
                val2 = state0[i][j-1] + gap_open
            if j > 0 and i > 0:
                val2 = max(val2, state2[i][j-1] + gap_extend)
            state2[i][j] = val2

    # Determine maximum score
    if mode == "global":
        max_score = state0[len_a][len_b]
        aligned_a, aligned_b = ",".join(["" for _ in range(len_a)]), ",".join(["" for _ in range(len_b)])
        i, j = len_a, len_b
        while i > 0 or j > 0:
            if i > 0 and j > 0 and state0[i][j] == state0[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif i > 0 and state1[i][j] == state1[i-1][j] + gap_extend:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            elif j > 0 and state2[i][j] == state2[i][j-1] + gap_extend:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
            else:
                break
        return (max_score, aligned_a, aligned_b)
    else:
        max_score = max(max(row) for row in state0 + state1 + state2)
        if max_score == 0:
            return (0, "", "")
        # Find the cell with max_score
        for i in range(len_a+1):
            for j in range(len_b+1):
                if state0[i][j] == max_score or state1[i][j] == max_score or state2[i][j] == max_score:
                    # Backtrack from here
                    aligned_a, aligned_b = "", ""
                    while i > 0 or j > 0:
                        if i > 0 and j > 0 and state0[i][j] == state0[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                            aligned_a = seq_a[i-1] + aligned_a
                            aligned_b = seq_b[j-1] + aligned_b
                            i -= 1
                            j -= 1
                        elif i > 0 and state1[i][j] == state1[i-1][j] + gap_extend:
                            aligned_a = seq_a[i-1] + aligned_a
                            aligned_b = "-" + aligned_b
                            i -= 1
                        elif j > 0 and state2[i][j] == state2[i][j-1] + gap_extend:
                            aligned_a = "-" + aligned_a
                            aligned_b = seq_b[j-1] + aligned_b
                            j -= 1
                        else:
                            break
                    return (max_score, aligned_a, aligned_b)
        return (max_score, "", "")

    # For local mode, return the aligned sequences
    return (max_score, aligned_a, aligned_b)