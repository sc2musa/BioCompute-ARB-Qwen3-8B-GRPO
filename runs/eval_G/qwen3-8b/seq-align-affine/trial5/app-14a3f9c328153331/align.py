import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    m = len(seq_a)
    n = len(seq_b)

    # Initialize DP tables
    dp = np.full((m+1, n+1), -np.inf)
    gap_a = np.full((m+1, n+1), -np.inf)
    gap_b = np.full((m+1, n+1), -np.inf)

    dp[0][0] = 0

    # Fill first row and column
    for j in range(1, n+1):
        gap_a[0][j] = gap_open + (j-1)*gap_extend
    for i in range(1, m+1):
        gap_b[i][0] = gap_open + (i-1)*gap_extend

    # Fill the rest of the table
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Compute current pair score
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch

            # Compute options for dp[i][j]
            option1 = dp[i-1][j-1] + score
            option2 = gap_a[i-1][j] + gap_extend
            option3 = gap_b[i-1][j] + gap_extend
            if mode == "local":
                dp[i][j] = max(option1, option2, option3, 0)
            else:
                dp[i][j] = max(option1, option2, option3)

            # Compute gap_a[i][j]
            gap_a[i][j] = max(gap_a[i-1][j] + gap_extend, dp[i-1][j] + gap_open)
            # Compute gap_b[i][j]
            gap_b[i][j] = max(gap_b[i-1][j] + gap_extend, dp[i-1][j] + gap_open)

    # Determine the optimal score and position
    if mode == "global":
        score = dp[m][n]
        i, j = m, n
    else:
        max_score = np.max(dp)
        if max_score <= 0:
            return (0, "", "")
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        score = max_score

    # Backtrack to find the alignment
    aligned_a = []
    aligned_b = []
    while i > 0 and j > 0:
        # Check if the current cell came from a match/mismatch
        if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif dp[i][j] == gap_a[i-1][j] + gap_extend:
            # Extending gap in a
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            i -= 1
        elif dp[i][j] == dp[i-1][j] + gap_open:
            # Starting new gap in a
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            i -= 1
        elif dp[i][j] == gap_b[i-1][j] + gap_extend:
            # Extending gap in b
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            j -= 1
        elif dp[i][j] == dp[i-1][j] + gap_open:
            # Starting new gap in b
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            j -= 1
        else:
            break

    # Handle remaining gaps
    while i > 0:
        aligned_a.append(seq_a[i-1])
        aligned_b.append("-")
        i -= 1
    while j > 0:
        aligned_a.append("-")
        aligned_b.append(seq_b[j-1])
        j -= 1

    # Reverse the aligned sequences
    aligned_a = ''.join(reversed(aligned_a))
    aligned_b = ''.join(reversed(aligned_b))

    return (score, aligned_a, aligned_b)