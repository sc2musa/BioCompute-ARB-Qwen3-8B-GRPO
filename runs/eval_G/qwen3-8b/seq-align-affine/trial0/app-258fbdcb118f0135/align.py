import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    if not seq_a and not seq_b:
        return (0, "", "")
    elif not seq_a:
        score = gap_open + (len(seq_b) - 1) * gap_extend
        return (score, seq_a, "-" * len(seq_b))
    elif not seq_b:
        score = gap_open + (len(seq_a) - 1) * gap_extend
        return (score, "-" * len(seq_a), seq_b)

    n, m = len(seq_a), len(seq_b)
    if mode == "local":
        # Smith-Waterman for local alignment
        dp = [[0]*(m+1) for _ in range(n+1)]
        gap_a = [[-np.inf]*(m+1) for _ in range(n+1)]
        gap_b = [[-np.inf]*(m+1) for _ in range(n+1)]
        max_score = -np.inf
        best_i, best_j = 0, 0
        for i in range(n+1):
            for j in range(m+1):
                if i == 0 and j == 0:
                    dp[i][j] = 0
                elif i == 0:
                    dp[i][j] = max(0, gap_open + (j-1)*gap_extend)
                    gap_a[i][j] = dp[i][j-1] + gap_extend
                elif j == 0:
                    dp[i][j] = max(0, gap_open + (i-1)*gap_extend)
                    gap_b[i][j] = dp[i-1][j] + gap_extend
                else:
                    if seq_a[i-1] == seq_b[j-1]:
                        score = match
                    else:
                        score = mismatch
                    dp[i][j] = max(
                        dp[i-1][j-1] + score,
                        gap_a[i-1][j] + gap_extend,
                        gap_b[i-1][j] + gap_extend,
                        0
                    )
                    gap_a[i][j] = max(
                        gap_a[i-1][j] + gap_extend,
                        dp[i-1][j] + gap_open
                    )
                    gap_b[i][j] = max(
                        gap_b[i][j-1] + gap_extend,
                        dp[i][j-1] + gap_open
                    )
                if dp[i][j] > max_score:
                    max_score = dp[i][j]
                    best_i, best_j = i, j
        if max_score <= 0:
            return (0, "", "")
        i, j = best_i, best_j
        aligned_a, aligned_b = [], []
        while i > 0 and j > 0:
            if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1]==seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif dp[i][j] == gap_a[i-1][j] + gap_extend:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif dp[i][j] == gap_b[i-1][j] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                j -= 1
            elif dp[i][j] == dp[i-1][j] + gap_open:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif dp[i][j] == dp[i][j-1] + gap_open:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                j -= 1
            else:
                break
        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        while j > 0:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
        return (max_score, aligned_a, aligned_b)
    else:
        # Needleman-Wunsch for global alignment
        dp = [[0]*(m+1) for _ in range(n+1)]
        gap_a = [[-np.inf]*(m+1) for _ in range(n+1)]
        gap_b = [[-np.inf]*(m+1) for _ in range(n+1)]
        for i in range(n+1):
            for j in range(m+1):
                if i == 0 and j == 0:
                    dp[i][j] = 0
                elif i == 0:
                    dp[i][j] = gap_open + (j-1)*gap_extend
                    gap_a[i][j] = dp[i][j-1] + gap_extend
                elif j == 0:
                    dp[i][j] = gap_open + (i-1)*gap_extend
                    gap_b[i][j] = dp[i-1][j] + gap_extend
                else:
                    if seq_a[i-1] == seq_b[j-1]:
                        score = match
                    else:
                        score = mismatch
                    dp[i][j] = max(
                        dp[i-1][j-1] + score,
                        gap_a[i-1][j] + gap_extend,
                        gap_b[i-1][j] + gap_extend
                    )
                    gap_a[i][j] = max(
                        gap_a[i-1][j] + gap_extend,
                        dp[i-1][j] + gap_open
                    )
                    gap_b[i][j] = max(
                        gap_b[i][j-1] + gap_extend,
                        dp[i][j-1] + gap_open
                    )
        i, j = n, m
        aligned_a, aligned_b = [], []
        while i > 0 and j > 0:
            if dp[i][j] == dp[i-1][j-1] + (match if seq_a[i-1]==seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif dp[i][j] == gap_a[i-1][j] + gap_extend:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif dp[i][j] == gap_b[i-1][j] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                j -= 1
            elif dp[i][j] == dp[i-1][j] + gap_open:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif dp[i][j] == dp[i][j-1] + gap_open:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                j -= 1
            else:
                break
        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        while j > 0:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
        return (dp[n][m], aligned_a, aligned_b)