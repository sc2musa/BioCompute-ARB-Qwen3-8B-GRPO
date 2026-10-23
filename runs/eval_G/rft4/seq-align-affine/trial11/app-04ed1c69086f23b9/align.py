def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    if not seq_a and not seq_b:
        return (0, "", "")
    elif not seq_a:
        score = gap_open + (len(seq_b)-1)*gap_extend
        if mode == "global":
            return (score, "-"*len(seq_b), seq_b)
        else:
            return (max(0, score), "-"*len(seq_b), seq_b)
    elif not seq_b:
        score = gap_open + (len(seq_a)-1)*gap_extend
        if mode == "global":
            return (score, seq_a, "-"*len(seq_a))
        else:
            return (max(0, score), seq_a, "-"*len(seq_a))
    else:
        m, n = len(seq_a), len(seq_b)
        if mode == "global":
            match = [[0]*(n+1) for _ in range(m+1)]
            gapA = [[0]*(n+1) for _ in range(m+1)]
            gapB = [[0]*(n+1) for _ in range(m+1)]
            for i in range(1, m+1):
                gapA[i][0] = gap_open + (i-1)*gap_extend
            for j in range(1, n+1):
                gapB[0][j] = gap_open + (j-1)*gap_extend
        else:
            match = [[-float('inf')]*(n+1) for _ in range(m+1)]
            gapA = [[-float('inf')]*(n+1) for _ in range(m+1)]
            gapB = [[-float('inf')]*(n+1) for _ in range(m+1)]
            match[0][0] = 0
        for i in range(1, m+1):
            for j in range(1, n+1):
                if seq_a[i-1] == seq_b[j-1]:
                    current_score = match
                else:
                    current_score = mismatch
                # Compute match[i][j]
                option1 = match[i-1][j-1] + current_score
                option2 = gapA[i-1][j] + gap_extend
                option3 = gapB[i-1][j] + gap_extend
                match[i][j] = max(option1, option2, option3)
                # Compute gapA[i][j]
                option1_gapA = gapA[i-1][j] + gap_extend
                option2_gapA = match[i-1][j] + gap_open + gap_extend
                option3_gapA = gapB[i-1][j] + gap_open + gap_extend
                gapA[i][j] = max(option1_gapA, option2_gapA, option3_gapA)
                # Compute gapB[i][j]
                option1_gapB = gapB[i-1][j-1] + gap_extend
                option2_gapB = match[i-1][j] + gap_open + gap_extend
                option3_gapB = gapA[i-1][j] + gap_open + gap_extend
                gapB[i][j] = max(option1_gapB, option2_gapB, option3_gapB)
        if mode == "global":
            max_score = match[m][n]
        else:
            max_score = -float('inf')
            for i in range(m+1):
                for j in range(n+1):
                    if match[i][j] > max_score:
                        max_score = match[i][j]
                    if gapA[i][j] > max_score:
                        max_score = gapA[i][j]
                    if gapB[i][j] > max_score:
                        max_score = gapB[i][j]
            if max_score < 0:
                return (0, "", "")
            max_score = max_score
        if mode == "global":
            i, j = m, n
        else:
            max_i, max_j = 0, 0
            for i in range(m+1):
                for j in range(n+1):
                    if match[i][j] == max_score:
                        max_i, max_j = i, j
                    if gapA[i][j] == max_score:
                        max_i, max_j = i, j
                    if gapB[i][j] == max_score:
                        max_i, max_j = i, j
            i, j = max_i, max_j
        aligned_a = []
        aligned_b = []
        while i > 0 and j > 0:
            if match[i][j] == match[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif match[i][j] == gapA[i-1][j] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            elif match[i][j] == gapB[i-1][j] + gap_extend:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
            elif gapA[i][j] == gapA[i-1][j] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            elif gapA[i][j] == match[i-1][j] + gap_open + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif gapA[i][j] == gapB[i-1][j] + gap_open + gap_extend:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
            elif gapB[i][j] == gapB[i][j-1] + gap_extend:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
            elif gapB[i][j] == match[i-1][j] + gap_open + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif gapB[i][j] == gapA[i-1][j] + gap_open + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            else:
                break
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
        return (max_score, aligned_a, aligned_b)


