import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    if not seq_a and not seq_b:
        return (0, "", "")
    elif not seq_a:
        return (gap_open + (len(seq_b)-1)*gap_extend, "-"*len(seq_b), seq_b)
    elif not seq_b:
        return (gap_open + (len(seq_a)-1)*gap_extend, seq_a, "-"*len(seq_a))

    len_a, len_b = len(seq_a), len(seq_b)

    if mode == "global":
        dp = np.full((len_a+1, len_b+1), -np.inf)
        gapA = np.full((len_a+1, len_b+1), -np.inf)
        gapB = np.full((len_a+1, len_b+1), -np.inf)
        dp[0][0] = 0

        for j in range(1, len_b+1):
            gapA[0][j] = gap_open + (j-1)*gap_extend
        for i in range(1, len_a+1):
            gapB[i][0] = gap_open + (i-1)*gap_extend

        for i in range(1, len_a+1):
            for j in range(1, len_b+1):
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                dp[i][j] = max(
                    dp[i-1][j-1] + score,
                    gapA[i-1][j] + gap_extend,
                    gapB[i][j-1] + gap_extend
                )
                gapA[i][j] = max(
                    gapA[i-1][j] + gap_extend,
                    dp[i-1][j] + gap_open
                )
                gapB[i][j] = max(
                    gapB[i][j-1] + gap_extend,
                    dp[i][j-1] + gap_open
                )

        final_score = dp[len_a][len_b]
        aligned_a, aligned_b = [], []
        i, j = len_a, len_b
        while i > 0 and j > 0:
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            if dp[i-1][j-1] + score == dp[i][j]:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif gapA[i-1][j] + gap_extend == dp[i][j]:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif gapB[i][j-1] + gap_extend == dp[i][j]:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                j -= 1
            else:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append('-')
            i -= 1
        while j > 0:
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            j -= 1
        aligned_a.reverse()
        aligned_b.reverse()
        return (final_score, ''.join(aligned_a), ''.join(aligned_b))
    elif mode == "local":
        dp = np.full((len_a+1, len_b+1), 0)
        gapA = np.full((len_a+1, len_b+1), -np.inf)
        gapB = np.full((len_a+1, len_b+1), -np.inf)
        for j in range(1, len_b+1):
            gapA[0][j] = gap_open + (j-1)*gap_extend
        for i in range(1, len_a+1):
            gapB[i][0] = gap_open + (i-1)*gap_extend
        for i in range(1, len_a+1):
            for j in range(1, len_b+1):
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                dp[i][j] = max(
                    dp[i-1][j-1] + score,
                    gapA[i-1][j] + gap_extend,
                    gapB[i][j-1] + gap_extend
                )
                dp[i][j] = max(dp[i][j], 0)
                gapA[i][j] = max(
                    gapA[i-1][j] + gap_extend,
                    dp[i-1][j] + gap_open
                )
                gapB[i][j] = max(
                    gapB[i][j-1] + gap_extend,
                    dp[i][j-1] + gap_open
                )
        max_score = np.max(dp)
        if max_score <= 0:
            return (0, "", "")
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        aligned_a, aligned_b = [], []
        while i > 0 and j > 0:
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            if dp[i-1][j-1] + score == dp[i][j]:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif gapA[i-1][j] + gap_extend == dp[i][j]:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif gapB[i][j-1] + gap_extend == dp[i][j]:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                j -= 1
            else:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append('-')
            i -= 1
        while j > 0:
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            j -= 1
        aligned_a.reverse()
        aligned_b.reverse()
        return (max_score, ''.join(aligned_a), ''.join(aligned_b))
    else:
        raise ValueError("Invalid mode")

if __name__ == "__main__":
    import sys
    seq_a, seq_b, mode, match, mismatch, gap_open, gap_extend = sys.argv[1:]
    match = int(match)
    mismatch = int(mismatch)
    gap_open = int(gap_open)
    gap_extend = int(gap_extend)
    print(align(seq_a, seq_b, mode, match, mismatch, gap_open, gap_extend))
