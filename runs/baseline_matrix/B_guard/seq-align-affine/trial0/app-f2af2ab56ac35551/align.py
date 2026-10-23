import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize scoring matrix
    len_a, len_b = len(seq_a), len(seq_b)
    score = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize first row and column based on mode
    if mode == "global":
        # Needleman-Wunsch style
        for i in range(len_a + 1):
            score[i][0] = i * gap_open
        for j in range(len_b + 1):
            score[0][j] = j * gap_open
    else:
        # Smith-Waterman style
        for i in range(len_a + 1):
            score[i][0] = 0
        for j in range(len_b + 1):
            score[0][j] = 0
    
    # Fill the scoring matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate the score for matching/mismatch
            match_score = score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            
            # Calculate the score for gap in seq_a
            gap_a_score = score[i-1][j] + gap_open
            if i > 1:
                gap_a_score = max(gap_a_score, score[i-2][j] + gap_extend)
            
            # Calculate the score for gap in seq_b
            gap_b_score = score[i][j-1] + gap_open
            if j > 1:
                gap_b_score = max(gap_b_score, score[i][j-2] + gap_extend)
            
            # Choose the maximum score
            score[i][j] = max(match_score, gap_a_score, gap_b_score)
    
    # Traceback to find the optimal alignment
    if mode == "global":
        # Needleman-Wunsch style
        i, j = len_a, len_b
        aligned_a, aligned_b = [], []
        while i > 0 or j > 0:
            if i > 0 and j > 0 and score[i][j] == score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif i > 0 and score[i][j] == score[i-1][j] + gap_open:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            else:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
        aligned_a = aligned_a[::-1]
        aligned_b = aligned_b[::-1]
    else:
        # Smith-Waterman style
        # Find the maximum score in the matrix
        max_score = np.max(score)
        if max_score <= 0:
            return (0, "", "")
        
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(score), score.shape)
        
        # Traceback to find the optimal alignment
        aligned_a, aligned_b = [], []
        while i > 0 and j > 0 and score[i][j] > 0:
            if i > 0 and j > 0 and score[i][j] == score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif i > 0 and score[i][j] == score[i-1][j] + gap_open:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            else:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
        aligned_a = aligned_a[::-1]
        aligned_b = aligned_b[::-1]
    
    # Calculate the score
    score_value = 0
    for a, b in zip(aligned_a, aligned_b):
        if a == "-" and b == "-":
            continue
        elif a == "-":
            score_value += gap_open + (len(aligned_a) - 1) * gap_extend
        elif b == "-":
            score_value += gap_open + (len(aligned_b) - 1) * gap_extend
        else:
            score_value += match if a == b else mismatch
    
    return (score_value, "".join(aligned_a), "".join(aligned_b))