import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    if not seq_a and not seq_b:
        return (0, "", "")
    elif not seq_a:
        return (0, "", seq_b)
    elif not seq_b:
        return (0, seq_a, "")

    len_a, len_b = len(seq_a), len(seq_b)
    INF = -10**18
    dp_match = np.full((len_a+1, len_b+1), INF, dtype=int)
    dp_gap_a = np.full((len_a+1, len_b+1), INF, dtype=int)
    dp_gap_b = np.full((len_a+1, len_b+1), INF, dtype=int)

    dp_match[0][0] = 0
    dp_gap_a[0][0] = INF
    dp_gap_b[0][0] = INF

    for i in range(len_a+1):
        for j in range(len_b+1):
            if i == 0 and j == 0:
                continue
            if i == 0:
                if j == 1:
                    dp_gap_b[i][j] = gap_open
                else:
                    dp_gap_b[i][j] = dp_gap_b[i][j-1] + gap_extend
                continue
            if j == 0:
                if i == 1:
                    dp_gap_a[i][j] = gap_open
                else:
                    dp_gap_a[i][j] = dp_gap_a[i-1][j] + gap_extend
                continue

            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch

            option1 = dp_match[i-1][j-1] + score
            option2 = dp_gap_a[i-1][j-1] + score
            option3 = dp_gap_b[i-1][j-1] + score
            dp_match[i][j] = max(option1, option2, option3)

            option1_gap_a = dp_match[i-1][j] + gap_open
            option2_gap_a = dp_gap_a[i-1][j] + gap_extend
            dp_gap_a[i][j] = max(option1_gap_a, option2_gap_a)

            option1_gap_b = dp_match[i][j-1] + gap_open
            option2_gap_b = dp_gap_b[i][j-1] + gap_extend
            dp_gap_b[i][j] = max(option1_gap_b, option2_gap_b)

    if mode == "global":
        score = dp_match[len_a][len_b]
    else:
        max_score = INF
        max_i, max_j = 0, 0
        for i in range(len_a+1):
            for j in range(len_b+1):
                current_max = max(dp_match[i][j], dp_gap_a[i][j], dp_gap_b[i][j])
                if current_max > max_score:
                    max_score = current_max
                    max_i, max_j = i, j
        if max_score <= 0:
            return (0, "", "")
        score = max_score

    # Backtrack
    i, j = (len_a, len_b) if mode == "global" else (max_i, max_j)
    aligned_a, aligned_b = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            # Check match
            if dp_match[i][j] == dp_match[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif dp_match[i][j] == dp_gap_a[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
                j -= 1
            elif dp_match[i][j] == dp_gap_b[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            else:
                # Fallback to gap checks
                pass
        elif i > 0:
            # Check gap_a
            if dp_gap_a[i][j] == dp_gap_a[i-1][j] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            elif dp_gap_a[i][j] == dp_match[i-1][j] + gap_open:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
        elif j > 0:
            # Check gap_b
            if dp_gap_b[i][j] == dp_gap_b[i][j-1] + gap_extend:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
            elif dp_gap_b[i][j] == dp_match[i][j-1] + gap_open:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
    return (score, ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))
