import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    # Initialize the DP table
    len_a = len(seq_a)
    len_b = len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1))

    # Fill the DP table
    for i in range(len_a + 1):
        for j in range(len_b + 1):
            if i == 0 and j == 0:
                dp[i, j] = 0
            elif i == 0:
                dp[i, j] = dp[i, j-1] + gap_open + (j-1)*gap_extend
            elif j == 0:
                dp[i, j] = dp[i-1, j] + gap_open + (i-1)*gap_extend
            else:
                match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                dp[i, j] = max(
                    dp[i-1, j-1] + match_score,
                    dp[i-1, j] + gap_open + (i-1)*gap_extend,
                    dp[i, j-1] + gap_open + (j-1)*gap_extend
                )

    # Backtrack to find the aligned sequences
    i, j = len_a, len_b
    aligned_a = []
    aligned_b = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            else:
                # Choose the best direction
                if dp[i-1, j-1] + match >= dp[i-1, j] + gap_open + (i-1)*gap_extend and dp[i-1, j-1] + match >= dp[i, j-1] + gap_open + (j-1)*gap_extend:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                elif dp[i-1, j] + gap_open + (i-1)*gap_extend >= dp[i, j-1] + gap_open + (j-1)*gap_extend:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                else:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    j -= 1
        elif i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append('-')
            i -= 1
        else:
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            j -= 1

    # Reverse the aligned sequences
    aligned_a = ''.join(reversed(aligned_a))
    aligned_b = ''.join(reversed(aligned_b))

    # Return the score and aligned sequences
    return (dp[len_a, len_b], aligned_a, aligned_b)