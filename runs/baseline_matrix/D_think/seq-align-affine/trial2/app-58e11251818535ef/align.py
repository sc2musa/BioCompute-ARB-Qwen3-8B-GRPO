def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """
    Return (score, aligned_a, aligned_b).
    """

    if not seq_a and not seq_b:
        return (0, "", "")
    elif not seq_a:
        # For global mode, the entire seq_b is aligned with gaps in seq_a
        # For local mode, the best substring of seq_b
        # But this is handled in the DP
        pass
    elif not seq_b:
        # Similar to above
        pass

    m = len(seq_a)
    n = len(seq_b)

    # Initialize DP tables
    M = [[0 for _ in range(n+1)] for _ in range(m+1)]
    G_a = [[0 for _ in range(n+1)] for _ in range(m+1)]
    G_b = [[0 for _ in range(n+1)] for _ in range(m+1)]

    # Initialize for global mode
    if mode == "global":
        # First row (seq_a is empty)
        for j in range(1, n+1):
            G_a[0][j] = gap_open + (j-1)*gap_extend
        # First column (seq_b is empty)
        for i in range(1, m+1):
            G_b[i][0] = gap_open + (i-1)*gap_extend
    else:  # local mode
        # All cells start with 0
        pass

    # Fill DP tables
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Compute match score
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch

            # Compute M[i][j]
            M[i][j] = max(
                M[i-1][j-1] + score,
                G_a[i-1][j-1] + score,
                G_b[i-1][j-1] + score
            )

            # Compute G_a[i][j]
            option1 = G_a[i-1][j] + gap_extend
            option2 = M[i-1][j] + gap_open
            option3 = G_b[i-1][j] + gap_open
            G_a[i][j] = max(option1, option2, option3)

            # Compute G_b[i][j]
            option1 = G_b[i][j-1] + gap_extend
            option2 = M[i][j-1] + gap_open
            option3 = G_a[i][j-1] + gap_open
            G_b[i][j] = max(option1, option2, option3)

    # Determine the final score
    if mode == "global":
        final_score = max(M[m][n], G_a[m][n], G_b[m][n])
    else:  # local
        max_score = 0
        for i in range(m+1):
            for j in range(n+1):
                current_max = max(M[i][j], G_a[i][j], G_b[i][j])
                if current_max > max_score:
                    max_score = current_max
        if max_score <= 0:
            return (0, "", "")
        final_score = max_score

    # Backtrack to find the alignment
    aligned_a = []
    aligned_b = []

    # For global mode
    if mode == "global":
        i, j = m, n
        while i > 0 and j > 0:
            # Determine which state contributed to the current cell
            current_max = max(M[i][j], G_a[i][j], G_b[i][j])
            if current_max == M[i][j]:
                # Match
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif current_max == G_a[i][j]:
                # Gap in seq_a
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
            else:  # G_b
                # Gap in seq_b
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                j -= 1
        # Handle remaining characters
        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append('-')
            i -= 1
        while j > 0:
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            j -= 1
        # Reverse the aligned sequences
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
    else:  # local mode
        # Find the cell with the maximum score
        max_score = 0
        max_i = 0
        max_j = 0
        for i in range(m+1):
            for j in range(n+1):
                current_max = max(M[i][j], G_a[i][j], G_b[i][j])
                if current_max > max_score:
                    max_score = current_max
                    max_i = i
                    max_j = j
        # Backtrack from max_i, max_j
        i, j = max_i, max_j
        while i > 0 and j > 0:
            current_max = max(M[i][j], G_a[i][j], G_b[i][j])
            if current_max == M[i][j]:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif current_max == G_a[i][j]:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
            else:  # G_b
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                j -= 1
        # Handle remaining characters
        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append('-')
            i -= 1
        while j > 0:
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            j -= 1
        # Reverse the aligned sequences
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))

    return (final_score, aligned_a, aligned_b)

