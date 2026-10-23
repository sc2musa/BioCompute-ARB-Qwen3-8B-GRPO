import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """
    Return (score, aligned_a, aligned_b).
    """
    if not seq_a and not seq_b:
        return (0, "", "")
    len_a, len_b = len(seq_a), len(seq_b)
    # Precompute score matrix
    score = np.zeros((len_a, len_b), dtype=np.int64)
    for i in range(len_a):
        for j in range(len_b):
            score[i, j] = match if seq_a[i] == seq_b[j] else mismatch
    # Global alignment
    if mode == "global":
        dp = np.full((len_a+1, len_b+1), -np.inf, dtype=np.int64)
        gap_a = np.full((len_a+1, len_b+1), -np.inf, dtype=np.int64)
        gap_b = np.full((len_a+1, len_b+1), -np.inf, dtype=np.int64)
        dp[0, 0] = 0
        # Initialize gap_b for first row
        for j in range(len_b+1):
            gap_b[0, j] = gap_open + (j-1)*gap_extend if j > 0 else 0
        # Initialize gap_a for first column
        for i in range(len_a+1):
            gap_a[i, 0] = gap_open + (i-1)*gap_extend if i > 0 else 0
        # Fill DP tables
        for i in range(1, len_a+1):
            for j in range(1, len_b+1):
                # Current score
                current_score = score[i-1, j-1]
                # Update dp[i, j]
                dp[i, j] = max(
                    dp[i-1, j-1] + current_score,
                    gap_a[i-1, j] + current_score,
                    gap_b[i-1, j] + current_score
                )
                # Update gap_a[i, j]
                gap_a[i, j] = max(
                    gap_a[i-1, j] + gap_extend,
                    dp[i-1, j] + gap_open
                )
                # Update gap_b[i, j]
                gap_b[i, j] = max(
                    gap_b[i, j-1] + gap_extend,
                    dp[i, j-1] + gap_open
                )
        # Backtrack
        aligned_a, aligned_b = [], []
        i, j = len_a, len_b
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                current_score = score[i-1, j-1]
                if dp[i, j] == dp[i-1, j-1] + current_score:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                elif dp[i, j] == gap_a[i-1, j] + current_score:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                elif dp[i, j] == gap_b[i-1, j] + current_score:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    j -= 1
            elif i > 0:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            elif j > 0:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
        return (dp[len_a, len_b], ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))
    # Local alignment
    elif mode == "local":
        dp = np.zeros((len_a+1, len_b+1), dtype=np.int64)
        gap_a = np.full((len_a+1, len_b+1), -np.inf, dtype=np.int64)
        gap_b = np.full((len_a+1, len_b+1), -np.inf, dtype=np.int64)
        # Initialize gap_b for first row
        for j in range(len_b+1):
            gap_b[0, j] = gap_open + (j-1)*gap_extend if j > 0 else 0
        # Initialize gap_a for first column
        for i in range(len_a+1):
            gap_a[i, 0] = gap_open + (i-1)*gap_extend if i > 0 else 0
        # Fill DP tables
        for i in range(1, len_a+1):
            for j in range(1, len_b+1):
                current_score = score[i-1, j-1]
                dp[i, j] = max(
                    dp[i-1, j-1] + current_score,
                    gap_a[i-1, j] + current_score,
                    gap_b[i-1, j] + current_score,
                    0
                )
                gap_a[i, j] = max(
                    gap_a[i-1, j] + gap_extend,
                    dp[i-1, j] + gap_open
                )
                gap_b[i, j] = max(
                    gap_b[i, j-1] + gap_extend,
                    dp[i, j-1] + gap_open
                )
        # Find max score
        max_score = dp.max()
        if max_score == 0:
            return (0, "", "")
        # Backtrack
        aligned_a, aligned_b = [], []
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                current_score = score[i-1, j-1]
                if dp[i, j] == dp[i-1, j-1] + current_score:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                elif dp[i, j] == gap_a[i-1, j] + current_score:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                elif dp[i, j] == gap_b[i-1, j] + current_score:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    j -= 1
            elif i > 0:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            elif j > 0:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
        return (max_score, ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))
    else:
        raise ValueError("Invalid mode")