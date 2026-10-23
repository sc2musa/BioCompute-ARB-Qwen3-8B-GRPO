import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    m, n = len(seq_a), len(seq_b)

    if m == 0 and n == 0:
        return (0, "", "")
    elif m == 0:
        return (0, "", seq_b)
    elif n == 0:
        return (0, seq_a, "")

    if mode == "global":
        match_dp = np.full((m+1, n+1), -float('inf'))
        gapA_dp = np.full((m+1, n+1), -float('inf'))
        gapB_dp = np.full((m+1, n+1), -float('inf'))
        match_dp[0][0] = 0
        for j in range(1, n+1):
            gapB_dp[0][j] = gap_open + (j-1)*gap_extend
        for i in range(1, m+1):
            gapA_dp[i][0] = gap_open + (i-1)*gap_extend
        for i in range(1, m+1):
            for j in range(1, n+1):
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                match_dp[i][j] = max(
                    match_dp[i-1][j-1] + score,
                    gapA_dp[i-1][j-1] + score,
                    gapB_dp[i-1][j-1] + score
                )
                gapA_dp[i][j] = max(
                    gapA_dp[i-1][j] + gap_extend,
                    match_dp[i-1][j] + gap_open
                )
                gapB_dp[i][j] = max(
                    gapB_dp[i][j-1] + gap_extend,
                    match_dp[i][j-1] + gap_open
                )
        score = max(match_dp[m][n], gapA_dp[m][n], gapB_dp[m][n])
        aligned_a, aligned_b = [], []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and match_dp[i][j] == score:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
                score -= (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            elif i > 0 and gapA_dp[i][j] == score:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
                score -= gap_extend
            elif j > 0 and gapB_dp[i][j] == score:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                j -= 1
                score -= gap_extend
            else:
                break
        return (score, ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))
    elif mode == "local":
        # Local mode implementation would go here
        pass
