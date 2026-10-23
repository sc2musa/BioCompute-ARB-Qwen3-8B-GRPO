def align(seq_a: str, seq_b: str, mode: str = "global",
              match: int = 2, mismatch: int = -3,
              gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)
    
    def match_score(a, b):
        return match if a == b else mismatch
    
    if mode == "global":
        # Global mode DP
        match_dp = [[-float("inf")]*(len_b+1) for _ in range(len_a+1)]
        gap_a_dp = [[-float("inf")]*(len_b+1) for _ in range(len_a+1)]
        gap_b_dp = [[-float("inf")]*(len_b+1) for _ in range(len_a+1)]
        
        match_dp[0][0] = 0
        for j in range(1, len_b+1):
            gap_a_dp[0][j] = gap_open + (j-1)*gap_extend
        for i in range(1, len_a+1):
            gap_b_dp[i][0] = gap_open + (i-1)*gap_extend
        
        for i in range(1, len_a+1):
            for j in range(1, len_b+1):
                score = match_score(seq_a[i-1], seq_b[j-1])
                match_dp[i][j] = max(
                    match_dp[i-1][j-1] + score,
                    gap_a_dp[i-1][j] + score,
                    gap_b_dp[i-1][j] + score
                )
                gap_a_dp[i][j] = max(
                    gap_a_dp[i-1][j] + gap_extend,
                    gap_b_dp[i-1][j] + gap_open + gap_extend,
                    match_dp[i-1][j] + gap_open + gap_extend
                )
                gap_b_dp[i][j] = max(
                    gap_b_dp[i][j-1] + gap_extend,
                    gap_a_dp[i][j-1] + gap_open + gap_extend,
                    match_dp[i][j-1] + gap_open + gap_extend
                )
        
        max_score = max(match_dp[len_a][len_b], gap_a_dp[len_a][len_b], gap_b_dp[len_a][len_b])
        
        # Backtrack
        aligned_a, aligned_b = [], []
        i, j = len_a, len_b
        if match_dp[i][j] >= gap_a_dp[i][j] and match_dp[i][j] >= gap_b_dp[i][j]:
            current_state = "match"
        elif gap_a_dp[i][j] >= match_dp[i][j] and gap_a_dp[i][j] >= gap_b_dp[i][j]:
            current_state = "gap_a"
        else:
            current_state = "gap_b"
        
        while i > 0 and j > 0:
            if current_state == "match":
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
                score = match_score(seq_a[i-1], seq_b[j-1])
                prev_match = match_dp[i-1][j-1] + score
                prev_gap_a = gap_a_dp[i-1][j] + score
                prev_gap_b = gap_b_dp[i-1][j] + score
                if match_dp[i][j] == prev_match:
                    current_state = "match"
                elif match_dp[i][j] == prev_gap_a:
                    current_state = "gap_a"
                else:
                    current_state = "gap_b"
            elif current_state == "gap_a":
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
                prev_gap_a = gap_a_dp[i][j] - gap_extend
                prev_gap_b = gap_b_dp[i][j] - (gap_open + gap_extend)
                prev_match = match_dp[i][j] - (gap_open + gap_extend)
                if gap_a_dp[i][j] == prev_gap_a:
                    current_state = "gap_a"
                elif gap_a_dp[i][j] == prev_gap_b:
                    current_state = "gap_b"
                else:
                    current_state = "match"
            else:  # gap_b
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                j -= 1
                prev_gap_b = gap_b_dp[i][j] - gap_extend
                prev_gap_a = gap_a_dp[i][j] - (gap_open + gap_extend)
                prev_match = match_dp[i][j] - (gap_open + gap_extend)
                if gap_b_dp[i][j] == prev_gap_b:
                    current_state = "gap_b"
                elif gap_b_dp[i][j] == prev_gap_a:
                    current_state = "gap_a"
                else:
                    current_state = "match"
        
        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        while j > 0:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
        
        aligned_a.reverse()
        aligned_b.reverse()
        return (max_score, "".join(aligned_a), "".join(aligned_b))
    
    elif mode == "local":
        # Local mode DP
        match_dp = [[-float("inf")]*(len_b+1) for _ in range(len_a+1)]
        gap_a_dp = [[-float("inf")]*(len_b+1) for _ in range(len_a+1)]
        gap_b_dp = [[-float("inf")]*(len_b+1) for _ in range(len_a+1)]
        
        match_dp[0][0] = 0
        
        for i in range(1, len_a+1):
            for j in range(1, len_b+1):
                score = match_score(seq_a[i-1], seq_b[j-1])
                match_dp[i][j] = max(
                    match_dp[i-1][j-1] + score,
                    gap_a_dp[i-1][j] + score,
                    gap_b_dp[i-1][j] + score,
                    0
                )
                gap_a_dp[i][j] = max(
                    gap_a_dp[i-1][j] + gap_extend,
                    gap_b_dp[i-1][j] + gap_open + gap_extend,
                    match_dp[i-1][j] + gap_open + gap_extend,
                    0
                )
                gap_b_dp[i][j] = max(
                    gap_b_dp[i][j-1] + gap_extend,
                    gap_a_dp[i][j-1] + gap_open + gap_extend,
                    match_dp[i][j-1] + gap_open + gap_extend,
                    0
                )
        
        max_score = 0
        for i in range(len_a+1):
            for j in range(len_b+1):
                max_score = max(max_score, match_dp[i][j], gap_a_dp[i][j], gap_b_dp[i][j])
        
        if max_score == 0:
            return (0, "", "")
        
        # Find starting cell
        i, j = 0, 0
        for x in range(len_a+1):
            for y in range(len_b+1):
                if match_dp[x][y] == max_score or gap_a_dp[x][y] == max_score or gap_b_dp[x][y] == max_score:
                    i, j = x, y
                    break
            else:
                continue
            break
        
        # Backtrack
        aligned_a, aligned_b = [], []
        current_state = "match" if match_dp[i][j] == max_score else ("gap_a" if gap_a_dp[i][j] == max_score else "gap_b")
        while i > 0 and j > 0:
            if current_state == "match":
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
                score = match_score(seq_a[i-1], seq_b[j-1])
                prev_match = match_dp[i-1][j-1] + score
                prev_gap_a = gap_a_dp[i-1][j] + score
                prev_gap_b = gap_b_dp[i-1][j] + score
                if match_dp[i][j] == prev_match:
                    current_state = "match"
                elif match_dp[i][j] == prev_gap_a:
                    current_state = "gap_a"
                else:
                    current_state = "gap_b"
            elif current_state == "gap_a":
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
                prev_gap_a = gap_a_dp[i][j] - gap_extend
                prev_gap_b = gap_b_dp[i][j] - (gap_open + gap_extend)
                prev_match = match_dp[i][j] - (gap_open + gap_extend)
                if gap_a_dp[i][j] == prev_gap_a:
                    current_state = "gap_a"
                elif gap_a_dp[i][j] == prev_gap_b:
                    current_state = "gap_b"
                else:
                    current_state = "match"
            else:  # gap_b
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                j -= 1
                prev_gap_b = gap_b_dp[i][j] - gap_extend
                prev_gap_a = gap_a_dp[i][j] - (gap_open + gap_extend)
                prev_match = match_dp[i][j] - (gap_open + gap_extend)
                if gap_b_dp[i][j] == prev_gap_b:
                    current_state = "gap_b"
                elif gap_b_dp[i][j] == prev_gap_a:
                    current_state = "gap_a"
                else:
                    current_state = "match"
        
        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        while j > 0:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
        
        aligned_a.reverse()
        aligned_b.reverse()
        return (max_score, "".join(aligned_a), "".join(aligned_b))
    
    else:
        raise ValueError("Invalid mode")