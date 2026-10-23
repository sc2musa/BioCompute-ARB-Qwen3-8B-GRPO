def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    # Handle empty sequences
    if not seq_a and not seq_b:
        return (0, "", "")
    if not seq_a:
        return (0, "", seq_b)
    if not seq_b:
        return (0, seq_a, "")

    # Initialize DP tables
    m, n = len(seq_a), len(seq_b)
    # For global alignment (Needleman-Wunsch)
    if mode == "global":
        # Initialize DP table with dimensions (m+1) x (n+1)
        dp = [[0]*(n+1) for _ in range(m+1)]
        # Initialize first row and column
        for i in range(m+1):
            dp[i][0] = i * gap_open + (i-1)*gap_extend if i > 0 else 0
        for j in range(n+1):
            dp[0][j] = j * gap_open + (j-1)*gap_extend if j > 0 else 0
        # Fill DP table
        for i in range(1, m+1):
            for j in range(1, n+1):
                # Match/mismatch score
                match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                # Score from diagonal (match/mismatch)
                diag = dp[i-1][j-1] + match_score
                # Score from up (gap in seq_b)
                up = dp[i-1][j] + gap_extend
                # Score from left (gap in seq_a)
                left = dp[i][j-1] + gap_extend
                # Choose maximum score
                dp[i][j] = max(diag, up, left)
        # Backtrack to find the optimal alignment
        i, j = m, n
        aligned_a, aligned_b = ",", ","
        while i > 0 and j > 0:
            # Check if we came from diagonal (match/mismatch)
            if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            # Check if we came from up (gap in seq_b)
            elif dp[i][j] == dp[i-1][j] + gap_extend:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            # Check if we came from left (gap in seq_a)
            elif dp[i][j] == dp[i][j-1] + gap_extend:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
            # Handle the case where we have a new gap
            else:
                # This should only happen if we have a new gap
                # Check if we came from up (gap in seq_b)
                if dp[i][j] == dp[i-1][j] + gap_open:
                    aligned_a = seq_a[i-1] + aligned_a
                    aligned_b = "-" + aligned_b
                    i -= 1
                # Check if we came from left (gap in seq_a)
                elif dp[i][j] == dp[i][j-1] + gap_open:
                    aligned_a = "-" + aligned_a
                    aligned_b = seq_b[j-1] + aligned_b
                    j -= 1
                # Handle the case where we have a new gap in both sequences
                # This should not happen with the current implementation
                else:
                    # This is a fallback case
                    aligned_a = "-" + aligned_a
                    aligned_b = "-" + aligned_b
                    i -= 1
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
        return (dp[m][n], aligned_a, aligned_b)
    # For local alignment (Smith-Waterman)
    elif mode == "local":
        # Initialize DP table with dimensions (m+1) x (n+1)
        dp = [[0]*(n+1) for _ in range(m+1)]
        # Fill DP table
        for i in range(1, m+1):
            for j in range(1, n+1):
                # Match/mismatch score
                match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                # Score from diagonal (match/mismatch)
                diag = dp[i-1][j-1] + match_score
                # Score from up (gap in seq_b)
                up = dp[i-1][j] + gap_extend
                # Score from left (gap in seq_a)
                left = dp[i][j-1] + gap_extend
                # Choose maximum score
                dp[i][j] = max(0, diag, up, left)
        # Find the maximum score in the DP table
        max_score = max(max(row) for row in dp)
        if max_score == 0:
            return (0, "", "")
        # Find the position of the maximum score
        i, j = 0, 0
        for x in range(m+1):
            for y in range(n+1):
                if dp[x][y] == max_score:
                    i, j = x, y
                    break
            else:
                continue
            break
        # Backtrack to find the optimal alignment
        aligned_a, aligned_b = ",", ","
        while i > 0 and j > 0:
            # Check if we came from diagonal (match/mismatch)
            if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            # Check if we came from up (gap in seq_b)
            elif dp[i][j] == dp[i-1][j] + gap_extend:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                i -= 1
            # Check if we came from left (gap in seq_a)
            elif dp[i][j] == dp[i][j-1] + gap_extend:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                j -= 1
            # Handle the case where we have a new gap
            else:
                # This should only happen if we have a new gap
                # Check if we came from up (gap in seq_b)
                if dp[i][j] == dp[i-1][j] + gap_open:
                    aligned_a = seq_a[i-1] + aligned_a
                    aligned_b = "-" + aligned_b
                    i -= 1
                # Check if we came from left (gap in seq_a)
                elif dp[i][j] == dp[i][j-1] + gap_open:
                    aligned_a = "-" + aligned_a
                    aligned_b = seq_b[j-1] + aligned_b
                    j -= 1
                # Handle the case where we have a new gap in both sequences
                # This should not happen with the current implementation
                else:
                    # This is a fallback case
                    aligned_a = "-" + aligned_a
                    aligned_b = "-" + aligned_b
                    i -= 1
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
    else:
        raise ValueError("Invalid mode. Use 'global' or 'local'.")