import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    if not seq_a and not seq_b:
        return (0, "", "")
    elif not seq_a:
        score = gap_open + (len(seq_b)-1)*gap_extend
        return (score, "-"*len(seq_b), seq_b)
    elif not seq_b:
        score = gap_open + (len(seq_a)-1)*gap_extend
        return (score, seq_a, "-"*len(seq_a))

    len_a, len_b = len(seq_a), len(seq_b)
    if mode == "global":
        dp = np.full((len_a+1, len_b+1), -np.inf)
        gap_a = np.full((len_a+1, len_b+1), -np.inf)
        gap_b = np.full((len_a+1, len_b+1), -np.inf)
        dp[0][0] = 0
        for j in range(1, len_b+1):
            gap_a[0][j] = gap_open + (j-1)*gap_extend
        for i in range(1, len_a+1):
            gap_b[i][0] = gap_open + (i-1)*gap_extend
    else:
        dp = np.zeros((len_a+1, len_b+1))
        gap_a = np.zeros((len_a+1, len_b+1))
        gap_b = np.zeros((len_a+1, len_b+1))
        for j in range(1, len_b+1):
            gap_a[0][j] = gap_open + (j-1)*gap_extend
        for i in range(1, len_a+1):
            gap_b[i][0] = gap_open + (i-1)*gap_extend

    for i in range(1, len_a+1):
        for j in range(1, len_b+1):
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            if mode == "global":
                dp[i][j] = max(dp[i-1][j-1] + score, gap_a[i-1][j] + score, gap_b[i-1][j] + score)
            else:
                dp[i][j] = max(dp[i-1][j-1] + score, gap_a[i-1][j] + score, gap_b[i-1][j] + score, 0)
            gap_a[i][j] = max(gap_a[i-1][j] + gap_extend, dp[i-1][j] + gap_open)
            gap_b[i][j] = max(gap_b[i-1][j] + gap_extend, dp[i-1][j] + gap_open)

    if mode == "global":
        final_score = dp[len_a][len_b]
        i, j = len_a, len_b
    else:
        final_score = np.max(dp)
        max_i, max_j = 0, 0
        for i in range(len_a+1):
            for j in range(len_b+1):
                if dp[i][j] > dp[max_i][max_j]:
                    max_i, max_j = i, j
        if final_score <= 0:
            return (0, "", "")
        i, j = max_i, max_j

    aligned_a, aligned_b = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            if dp[i][j] == dp[i-1][j-1] + score:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif dp[i][j] == gap_a[i-1][j] + score:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
            else:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                j -= 1
        elif i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        else:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
    aligned_a.reverse()
    aligned_b.reverse()
    return (final_score, ''.join(aligned_a), ''.join(aligned_b))