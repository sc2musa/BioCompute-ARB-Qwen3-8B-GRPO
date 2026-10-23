def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)
    
    # Initialize DP tables with a very small integer
    INF = -10**18
    match_table = [[INF] * (len_b + 1) for _ in range(len_a + 1)]
    gapA_table = [[INF] * (len_b + 1) for _ in range(len_a + 1)]
    gapB_table = [[INF] * (len_b + 1) for _ in range(len_a + 1)]
    
    # Global mode initialization
    if mode == "global":
        match_table[0][0] = 0
        for j in range(1, len_b + 1):
            gapA_table[0][j] = gap_open + (j - 1) * gap_extend
        for i in range(1, len_a + 1):
            gapB_table[i][0] = gap_open + (i - 1) * gap_extend
    else:  # Local mode
        match_table[0][0] = 0
    
    # Fill DP tables
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Match/mismatch score
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Match state
            match_val = max(
                match_table[i-1][j-1] + score,
                gapA_table[i-1][j] + score,
                gapB_table[i-1][j-1] + score
            )
            match_table[i][j] = match_val
            
            # Gap in A
            gapA_val = max(
                gapA_table[i-1][j] + gap_extend,
                match_table[i-1][j] + gap_open
            )
            gapA_table[i][j] = gapA_val
            
            # Gap in B
            gapB_val = max(
                gapB_table[i][j-1] + gap_extend,
                match_table[i][j-1] + gap_open
            )
            gapB_table[i][j] = gapB_val
    
    # Determine final score
    if mode == "global":
        final_score = max(match_table[len_a][len_b], gapA_table[len_a][len_b], gapB_table[len_a][len_b])
    else:
        max_score = INF
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                current = max(match_table[i][j], gapA_table[i][j], gapB_table[i][j])
                if current > max_score:
                    max_score = current
        if max_score <= 0:
            return (0, "", "")
        final_score = max_score
    
    # Backtrack to find alignment
    aligned_a, aligned_b = [], []
    if mode == "local":
        # Find maximum score position
        max_i, max_j = 0, 0
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                current = max(match_table[i][j], gapA_table[i][j], gapB_table[i][j])
                if current > max_score:
                    max_score = current
                    max_i, max_j = i, j
        i, j = max_i, max_j
    else:
        i, j = len_a, len_b
    
    # Backtrack while i or j > 0
    while i > 0 or j > 0:
        # Determine current state
        current_match = match_table[i][j]
        current_gapA = gapA_table[i][j]
        current_gapB = gapB_table[i][j]
        
        # Find which state contributed to the current value
        current_val = max(current_match, current_gapA, current_gapB)
        
        if current_val == current_match:
            # Match state
            # Check which previous state contributed to this match_val
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            prev_match = match_table[i-1][j-1] + score
            prev_gapA = gapA_table[i-1][j] + score
            prev_gapB = gapB_table[i-1][j-1] + score
            if current_val == prev_match:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif current_val == prev_gapA:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif current_val == prev_gapB:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                j -= 1
        elif current_val == current_gapA:
            # Gap in A
            prev_gapA = gapA_table[i-1][j] + gap_extend
            prev_match = match_table[i-1][j] + gap_open
            if current_val == prev_gapA:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif current_val == prev_match:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                j -= 1
        elif current_val == current_gapB:
            # Gap in B
            prev_gapB = gapB_table[i][j-1] + gap_extend
            prev_match = match_table[i][j-1] + gap_open
            if current_val == prev_gapB:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                j -= 1
            elif current_val == prev_match:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
        else:
            # Handle edge cases where i or j is 0
            if i == 0:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
            elif j == 0:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            else:
                # Default case, should not occur
                pass
    
    # Reverse to get correct order
    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]
    
    return (final_score, "".join(aligned_a), "".join(aligned_b))