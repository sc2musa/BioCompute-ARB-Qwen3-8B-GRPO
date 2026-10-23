import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Handle empty sequences
    if not seq_a and not seq_b:
        return (0, "", "")
    if not seq_a:
        return (0, "", seq_b)
    if not seq_b:
        return (0, seq_a, "")
    
    # Initialize the score matrix
    len_a, len_b = len(seq_a), len(seq_b)
    score = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize the first row and column based on the mode
    if mode == "global":
        # Global alignment (Needleman-Wunsch)
        # First column: gaps in seq_a
        for i in range(1, len_a + 1):
            score[i][0] = score[i-1][0] + gap_extend
        # First row: gaps in seq_b
        for j in range(1, len_b + 1):
            score[0][j] = score[0][j-1] + gap_extend
    else:
        # Local alignment (Smith-Waterman)
        # Initialize with 0s
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                score[i][j] = 0
    
    # Fill the score matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Match/mismatch score
            match_score = score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            
            # Gap in seq_a
            gap_a_score = score[i-1][j] + gap_extend
            
            # Gap in seq_b
            gap_b_score = score[i][j-1] + gap_extend
            
            # Choose the maximum score
            if mode == "global":
                score[i][j] = max(match_score, gap_a_score, gap_b_score)
            else:
                # For local alignment, we take the max of 0 and the other scores
                score[i][j] = max(0, match_score, gap_a_score, gap_b_score)
    
    # Traceback to find the optimal alignment
    # Find the optimal starting position
    if mode == "global":
        i, j = len_a, len_b
    else:
        # Find the maximum score in the matrix
        max_score = score.max()
        # Find the position of the maximum score
        i, j = np.unravel_index(score.argmax(), score.shape)
        # If the maximum score is 0, return (0, "", "")
        if max_score == 0:
            return (0, "", "")
    
    # Initialize the aligned sequences
    aligned_a = []
    aligned_b = []
    
    # Traceback
    while i > 0 and j > 0:
        # Check if the current cell came from a match/mismatch
        if score[i][j] == score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        # Check if the current cell came from a gap in seq_a
        elif score[i][j] == score[i-1][j] + gap_extend:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            i -= 1
        # Check if the current cell came from a gap in seq_b
        elif score[i][j] == score[i][j-1] + gap_extend:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            j -= 1
        else:
            # This should not happen
            break
    
    # Reverse the aligned sequences
    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]
    
    # Calculate the score
    score_value = 0
    # Track if we have seen a gap in each sequence
    gap_a_seen = False
    gap_b_seen = False
    
    for a, b in zip(aligned_a, aligned_b):
        if a == "-" and b == "-":
            continue
        elif a == "-":
            # Gap in seq_a
            if not gap_a_seen:
                score_value += gap_open
                gap_a_seen = True
            score_value += gap_extend
        elif b == "-":
            # Gap in seq_b
            if not gap_b_seen:
                score_value += gap_open
                gap_b_seen = True
            score_value += gap_extend
        else:
            # Match/mismatch
            score_value += (match if a == b else mismatch)
    
    return (score_value, "".join(aligned_a), "".join(aligned_b))