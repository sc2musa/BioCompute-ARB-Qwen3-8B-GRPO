import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize score matrix
    len_a, len_b = len(seq_a), len(seq_b)
    score = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize first row and column based on mode
    if mode == "global":
        # Needleman-Wunsch style
        for i in range(1, len_a + 1):
            score[i][0] = score[i-1][0] + gap_open
        for j in range(1, len_b + 1):
            score[0][j] = score[0][j-1] + gap_open
    elif mode == "local":
        # Smith-Waterman style
        for i in range(1, len_a + 1):
            score[i][0] = 0
        for j in range(1, len_b + 1):
            score[0][j] = 0
    else:
        raise ValueError("Invalid mode. Use 'global' or 'local'.")
    
    # Fill the score matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Match/mismatch score
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Gap scores
            gap_score_a = score[i-1][j] + gap_extend
            gap_score_b = score[i][j-1] + gap_extend
            
            # Choose the maximum score
            score[i][j] = max(
                score[i-1][j-1] + match_score,
                gap_score_a,
                gap_score_b
            )
    
    # Determine the maximum score based on mode
    if mode == "global":
        max_score = score[len_a][len_b]
    elif mode == "local":
        max_score = np.max(score)
    else:
        raise ValueError("Invalid mode. Use 'global' or 'local'.")
    
    # Backtrack to find the optimal alignment
    aligned_a = []
    aligned_b = []
    i, j = len_a, len_b
    
    while i > 0 or j > 0:
        if mode == "global":
            if i > 0 and j > 0:
                if score[i][j] == score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                elif i > 0 and score[i][j] == score[i-1][j] + gap_extend:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    i -= 1
                elif j > 0 and score[i][j] == score[i][j-1] + gap_extend:
                    aligned_a.append("-")
                    aligned_b.append(seq_b[j-1])
                    j -= 1
            else:
                # Handle remaining gaps
                if i > 0:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    i -= 1
                elif j > 0:
                    aligned_a.append("-")
                    aligned_b.append(seq_b[j-1])
                    j -= 1
        elif mode == "local":
            if i > 0 and j > 0:
                if score[i][j] == score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                elif i > 0 and score[i][j] == score[i-1][j] + gap_extend:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    i -= 1
                elif j > 0 and score[i][j] == score[i][j-1] + gap_extend:
                    aligned_a.append("-")
                    aligned_b.append(seq_b[j-1])
                    j -= 1
            else:
                # Handle remaining gaps
                if i > 0:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    i -= 1
                elif j > 0:
                    aligned_a.append("-")
                    aligned_b.append(seq_b[j-1])
                    j -= 1
                else:
                    break
    
    # Reverse the aligned sequences
    aligned_a = ''.join(reversed(aligned_a))
    aligned_b = ''.join(reversed(aligned_b))
    
    # Return the result
    return (max_score, aligned_a, aligned_b)