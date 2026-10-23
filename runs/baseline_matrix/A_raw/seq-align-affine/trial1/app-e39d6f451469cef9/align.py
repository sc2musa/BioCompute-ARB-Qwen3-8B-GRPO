def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    import numpy as np

    # Handle empty sequences
    if not seq_a and not seq_b:
        return (0, "", "")
    if not seq_a:
        return (0, "", seq_b)
    if not seq_b:
        return (0, seq_a, "")

    # Initialize DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)

    # Initialize first row and column
    for i in range(len_a + 1):
        dp[i][0] = i * gap_open + (i - 1) * gap_extend if i > 0 else 0
    for j in range(len_b + 1):
        dp[0][j] = j * gap_open + (j - 1) * gap_extend if j > 0 else 0

    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Match/mismatch score
            match_score = dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            # Gap in seq_a
            gap_a = dp[i-1][j] + gap_open + gap_extend * (i - 1)
            # Gap in seq_b
            gap_b = dp[i][j-1] + gap_open + gap_extend * (j - 1)
            # Choose the maximum score
            dp[i][j] = max(match_score, gap_a, gap_b)

    # Traceback to find the alignment
    i, j = len_a, len_b
    aligned_a, aligned_b = "", ""
    while i > 0 and j > 0:
        if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j] + gap_open + gap_extend * (i - 1):
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        else:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1

    # Handle remaining characters
    while i > 0:
        aligned_a = seq_a[i-1] + aligned_a
        aligned_b = "-" + aligned_b
        i -= 1
    while j > 0:
        aligned_a = "-" + aligned_a
        aligned_b = seq_b[j-1] + aligned_b
        j -= 1

    # Return the score and alignments
    if mode == "global":
        return (dp[len_a][len_b], aligned_a, aligned_b)
    else:
        # Local mode: find the maximum score in the DP table
        max_score = dp.max()
        if max_score <= 0:
            return (0, "", "")
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        # Traceback from the maximum score
        aligned_a, aligned_b = "", ""
        while i > 0 and j > 0:
            if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i-1][j] + gap_open + gap_extend * (i - 1):
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            else:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
        # Handle remaining characters
        while i > 0:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        while j > 0:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
        return (max_score, aligned_a, aligned_b)