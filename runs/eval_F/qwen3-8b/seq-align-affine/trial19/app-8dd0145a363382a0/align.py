import numpy as np


def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the score matrix
    m, n = len(seq_a), len(seq_b)
    score = np.zeros((m + 1, n + 1), dtype=int)
    
    # Initialize the first row and column with affine gap penalties
    for i in range(1, m + 1):
        score[i][0] = score[i-1][0] + gap_open + (i - 1) * gap_extend
    for j in range(1, n + 1):
        score[0][j] = score[0][j-1] + gap_open + (j - 1) * gap_extend
    
    # Fill the score matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_score = score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            delete_score = score[i-1][j] + gap_extend
            insert_score = score[i][j-1] + gap_extend
            score[i][j] = max(match_score, delete_score, insert_score)
    
    # Traceback to find the optimal alignment
    i, j = m, n
    aligned_a, aligned_b = [], []
    
    while i > 0 and j > 0:
        if seq_a[i-1] == seq_b[j-1]:
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif score[i][j] == score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif score[i][j] == score[i-1][j] + gap_extend:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        else:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
    
    # Reverse the alignment
    aligned_a = ''.join(reversed(aligned_a))
    aligned_b = ''.join(reversed(aligned_b))
    
    # Handle local mode
    if mode == "local":
        # Find the maximum score in the matrix
        max_score = np.max(score)
        # If no positive-scoring local alignment exists
        if max_score <= 0:
            return (0, "", "")
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(score), score.shape)
        # Traceback from the maximum score
        aligned_a, aligned_b = [], []
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif score[i][j] == score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif score[i][j] == score[i-1][j] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            else:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
        # Reverse the alignment
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
        # Trim the alignment to the local region
        start_a = aligned_a.find("-")
        start_b = aligned_b.find("-")
        end_a = aligned_a.rfind("-")
        end_b = aligned_b.rfind("-")
        if start_a == -1 and start_b == -1:
            # No gaps in the alignment
            aligned_a = aligned_a
            aligned_b = aligned_b
        else:
            # Trim the alignment to the local region
            aligned_a = aligned_a[start_a:end_a+1]
            aligned_b = aligned_b[start_b:end_b+1]
    
    # Return the score and aligned sequences
    return (score[m][n], aligned_a, aligned_b)