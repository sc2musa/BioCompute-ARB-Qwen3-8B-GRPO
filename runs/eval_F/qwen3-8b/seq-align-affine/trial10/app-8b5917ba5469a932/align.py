import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the score matrix
    len_a, len_b = len(seq_a), len(seq_b)
    score = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize the first row and column based on the mode
    if mode == "global":
        for i in range(1, len_a + 1):
            score[i][0] = score[i-1][0] + gap_open
        for j in range(1, len_b + 1):
            score[0][j] = score[0][j-1] + gap_open
    else:  # local mode
        for i in range(1, len_a + 1):
            score[i][0] = 0
        for j in range(1, len_b + 1):
            score[0][j] = 0
    
    # Fill the score matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            score[i][j] = max(
                score[i-1][j-1] + match_score,
                score[i-1][j] + gap_open,  # gap in seq_a
                score[i][j-1] + gap_open   # gap in seq_b
            )
    
    # Determine the optimal score based on the mode
    if mode == "global":
        score_val = score[len_a][len_b]
    else:  # local mode
        score_val = np.max(score)
    
    # Traceback to find the optimal alignment
    aligned_a = []
    aligned_b = []
    i, j = len_a, len_b
    
    while i > 0 and j > 0:
        if score[i][j] == score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif score[i][j] == score[i-1][j] + gap_open:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        else:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
    
    # Reverse the aligned sequences
    aligned_a = ''.join(reversed(aligned_a))
    aligned_b = ''.join(reversed(aligned_b))
    
    return (score_val, aligned_a, aligned_b)