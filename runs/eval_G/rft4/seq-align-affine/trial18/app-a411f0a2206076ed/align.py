def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    # Handle empty sequences
    if not seq_a and not seq_b:
        return (0, "", "")
    if not seq_a:
        return (gap_open + (len(seq_b)-1)*gap_extend, "-"*len(seq_b), seq_b)
    if not seq_b:
        return (gap_open + (len(seq_a)-1)*gap_extend, seq_a, "-"*len(seq_a))

    len_a, len_b = len(seq_a), len(seq_b)

    # Initialize DP tables
    main = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
    gap_a = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
    gap_b = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]

    # Global mode
    if mode == "global":
        main[0][0] = 0
        for j in range(1, len_b + 1):
            gap_b[0][j] = gap_open + (j-1)*gap_extend
        for i in range(1, len_a + 1):
            gap_a[i][0] = gap_open + (i-1)*gap_extend

        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                # Compute main[i][j]
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                main[i][j] = max(
                    main[i-1][j-1] + score,
                    gap_a[i-1][j] + gap_extend,
                    gap_b[i-1][j] + gap_extend
                )
                # Compute gap_a[i][j]
                gap_a[i][j] = max(
                    gap_a[i-1][j] + gap_extend,
                    main[i-1][j] + gap_open
                )
                # Compute gap_b[i][j]
                gap_b[i][j] = max(
                    gap_b[i-1][j] + gap_extend,
                    main[i-1][j] + gap_open
                )
        score = main[len_a][len_b]
        # Backtrack
        aligned_a, aligned_b = "", ""
        i, j = len_a, len_b
        while i > 0 and j > 0:
            if main[i][j] == main[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a += seq_a[i-1]
                aligned_b += seq_b[j-1]
                i -= 1
                j -= 1
            elif main[i][j] == gap_a[i-1][j] + gap_extend:
                aligned_a += seq_a[i-1]
                aligned_b += '-'
                i -= 1
            elif main[i][j] == gap_b[i-1][j] + gap_extend:
                aligned_a += '-'
                aligned_b += seq_b[j-1]
                j -= 1
            else:
                # This should not happen
                break
        # Handle remaining gaps
        while i > 0:
            aligned_a += seq_a[i-1]
            aligned_b += '-'
            i -= 1
        while j > 0:
            aligned_a += '-'
            aligned_b += seq_b[j-1]
            j -= 1
        return (score, aligned_a[::-1], aligned_b[::-1])

    # Local mode
    elif mode == "local":
        # Initialize DP tables with 0s
        main = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        gap_a = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        gap_b = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        main[0][0] = 0
        # Initialize first row and column
        for j in range(1, len_b + 1):
            main[0][j] = max(0, gap_open + (j-1)*gap_extend)
        for i in range(1, len_a + 1):
            main[i][0] = max(0, gap_open + (i-1)*gap_extend)
        # Fill DP tables
        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                main[i][j] = max(
                    main[i-1][j-1] + score,
                    gap_a[i-1][j] + gap_extend,
                    gap_b[i-1][j] + gap_extend,
                    0
                )
                gap_a[i][j] = max(
                    gap_a[i-1][j] + gap_extend,
                    main[i-1][j] + gap_open
                )
                gap_b[i][j] = max(
                    gap_b[i-1][j] + gap_extend,
                    main[i-1][j] + gap_open
                )
        # Find the maximum score
        max_score = max(max(row) for row in main)
        if max_score <= 0:
            return (0, "", "")
        # Find the position of the maximum score
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if main[i][j] == max_score:
                    start_i, start_j = i, j
                    break
            else:
                continue
            break
        # Backtrack from start_i, start_j
        aligned_a, aligned_b = "", ""
        i, j = start_i, start_j
        while i > 0 and j > 0:
            if main[i][j] == main[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a += seq_a[i-1]
                aligned_b += seq_b[j-1]
                i -= 1
                j -= 1
            elif main[i][j] == gap_a[i-1][j] + gap_extend:
                aligned_a += seq_a[i-1]
                aligned_b += '-'
                i -= 1
            elif main[i][j] == gap_b[i-1][j] + gap_extend:
                aligned_a += '-'
                aligned_b += seq_b[j-1]
                j -= 1
            else:
                # This should not happen
                break
            # Stop if we reach the start of the alignment
            if main[i][j] == 0:
                break
        return (max_score, aligned_a[::-1], aligned_b[::-1])

    # Handle invalid mode
    raise ValueError("Invalid mode. Use 'global' or 'local'.")