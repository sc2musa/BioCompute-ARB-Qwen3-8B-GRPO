import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)
    if len_a == 0 and len_b == 0:
        return (0, "", "")

    # Initialize DP tables
    if mode == "global":
        dp = np.full((len_a+1, len_b+1), -float('inf'))
        gap_a = np.full((len_a+1, len_b+1), -float('inf'))
        gap_b = np.full((len_a+1, len_b+1), -float('inf'))
        dp[0, 0] = 0
        for j in range(1, len_b+1):
            gap_a[0, j] = gap_open + (j-1)*gap_extend
            gap_b[0, j] = gap_open + (j-1)*gap_extend
        for i in range(1, len_a+1):
            gap_a[i, 0] = gap_open + (i-1)*gap_extend
            gap_b[i, 0] = gap_open + (i-1)*gap_extend
        
        for i in range(1, len_a+1):
            for j in range(1, len_b+1):
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                dp[i, j] = max(
                    dp[i-1, j-1] + score,
                    gap_a[i-1, j] + score,
                    gap_b[i, j-1] + score
                )
                gap_a[i, j] = max(
                    gap_a[i-1, j] + gap_extend,
                    dp[i-1, j] + gap_open
                )
                gap_b[i, j] = max(
                    gap_b[i, j-1] + gap_extend,
                    dp[i, j-1] + gap_open
                )
        score = dp[len_a, len_b]
        aligned_a, aligned_b = backtrack_global(dp, gap_a, gap_b, seq_a, seq_b, len_a, len_b, match, mismatch, gap_open, gap_extend)
        return (score, aligned_a, aligned_b)
    elif mode == "local":
        dp = np.full((len_a+1, len_b+1), -float('inf'))
        gap_a = np.full((len_a+1, len_b+1), -float('inf'))
        gap_b = np.full((len_a+1, len_b+1), -float('inf'))
        dp[0, 0] = 0
        
        for i in range(len_a+1):
            for j in range(len_b+1):
                if i > 0 and j > 0:
                    score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                else:
                    score = 0
                dp[i, j] = max(
                    dp[i-1, j-1] + score if i > 0 and j > 0 else -float('inf'),
                    gap_a[i-1, j] + score if i > 0 else -float('inf'),
                    gap_b[i, j-1] + score if j > 0 else -float('inf')
                )
                gap_a[i, j] = max(
                    gap_a[i-1, j] + gap_extend if i > 0 else -float('inf'),
                    dp[i-1, j] + gap_open if i > 0 else -float('inf')
                )
                gap_b[i, j] = max(
                    gap_b[i, j-1] + gap_extend if j > 0 else -float('inf'),
                    dp[i, j-1] + gap_open if j > 0 else -float('inf')
                )
        
        max_score = np.max([dp.flatten(), gap_a.flatten(), gap_b.flatten()])
        if max_score <= 0:
            return (0, "", "")
        
        for i in range(len_a+1):
            for j in range(len_b+1):
                if dp[i, j] == max_score:
                    current_i, current_j = i, j
                    break
                elif gap_a[i, j] == max_score:
                    current_i, current_j = i, j
                    break
                elif gap_b[i, j] == max_score:
                    current_i, current_j = i, j
                    break
        
        aligned_a, aligned_b = backtrack_local(dp, gap_a, gap_b, seq_a, seq_b, current_i, current_j, match, mismatch, gap_open, gap_extend)
        return (max_score, aligned_a, aligned_b)
    else:
        raise ValueError("Invalid mode")


def backtrack_global(dp, gap_a, gap_b, seq_a, seq_b, len_a, len_b, match, mismatch, gap_open, gap_extend):
    aligned_a = []
    aligned_b = []
    i, j = len_a, len_b
    while i > 0 and j > 0:
        if dp[i, j] == dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif gap_a[i, j] == gap_a[i-1, j] + gap_extend:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            i -= 1
        elif gap_b[i, j] == gap_b[i, j-1] + gap_extend:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            j -= 1
    while i > 0:
        aligned_a.append(seq_a[i-1])
        aligned_b.append("-")
        i -= 1
    while j > 0:
        aligned_a.append("-")
        aligned_b.append(seq_b[j-1])
        j -= 1
    return ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b))

def backtrack_local(dp, gap_a, gap_b, seq_a, seq_b, i, j, match, mismatch, gap_open, gap_extend):
    aligned_a = []
    aligned_b = []
    while i > 0 and j > 0:
        if dp[i, j] == dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif gap_a[i, j] == gap_a[i-1, j] + gap_extend:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            i -= 1
        elif gap_b[i, j] == gap_b[i, j-1] + gap_extend:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            j -= 1
    while i > 0:
        aligned_a.append(seq_a[i-1])
        aligned_b.append("-")
        i -= 1
    while j > 0:
        aligned_a.append("-")
        aligned_b.append(seq_b[j-1])
        j -= 1
    return ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b))
