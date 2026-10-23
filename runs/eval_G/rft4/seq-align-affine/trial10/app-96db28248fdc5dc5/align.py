import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    m, n = len(seq_a), len(seq_b)
    if m == 0 or n == 0:
        if mode == "global":
            return (0, "-" * n, seq_b) if m == 0 else (0, seq_a, "-" * m)
        else:
            return (0, "", "")

    # DP table for global/local
    if mode == "global":
        dp = np.full((m+1, n+1), -np.inf)
        dp[0, 0] = 0
        # Initialize first row and column
        for i in range(1, m+1):
            dp[i, 0] = dp[i-1, 0] + gap_open
        for j in range(1, n+1):
            dp[0, j] = dp[0, j-1] + gap_open
    else:  # local
        dp = np.full((m+1, n+1), -np.inf)
        dp[0, 0] = 0
        for i in range(1, m+1):
            dp[i, 0] = max(dp[i-1, 0] + gap_open, 0)
            dp[0, i] = max(dp[0, i-1] + gap_open, 0)

    # Fill DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Match/mismatch
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            dp[i, j] = dp[i-1, j-1] + score
            
            # Gap in a
            if mode == "global":
                dp[i, j] = max(dp[i, j], dp[i-1, j] + gap_open)
            else:
                dp[i, j] = max(dp[i, j], dp[i-1, j] + gap_open, 0)
            
            # Gap in b
            if mode == "global":
                dp[i, j] = max(dp[i, j], dp[i, j-1] + gap_open)
            else:
                dp[i, j] = max(dp[i, j], dp[i, j-1] + gap_open, 0)
            
            # Extend gap in a
            if mode == "global":
                dp[i, j] = max(dp[i, j], dp[i-1, j] + gap_extend)
            else:
                dp[i, j] = max(dp[i, j], dp[i-1, j] + gap_extend, 0)
            
            # Extend gap in b
            if mode == "global":
                dp[i, j] = max(dp[i, j], dp[i, j-1] + gap_extend)
            else:
                dp[i, j] = max(dp[i, j], dp[i, j-1] + gap_extend, 0)

    # Extract score and alignment
    if mode == "global":
        score = dp[m, n]
        aligned_a, aligned_b = "", ""
        i, j = m, n
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif dp[i, j] == dp[i-1, j] + gap_open:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            elif dp[i, j] == dp[i, j-1] + gap_open:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
            else:
                aligned_a = "-" + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
                j -= 1
        while i > 0:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        while j > 0:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
    else:
        score = np.max(dp)
        if score <= 0:
            return (0, "", "")
        # Find the cell with max score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        aligned_a, aligned_b = "", ""
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif dp[i, j] == dp[i-1, j] + gap_open:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            elif dp[i, j] == dp[i, j-1] + gap_open:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
            else:
                aligned_a = "-" + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
                j -= 1
        while i > 0:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        while j > 0:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
    return (score, aligned_a, aligned_b)