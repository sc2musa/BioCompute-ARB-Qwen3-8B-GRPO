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
    
    # Initialize score matrix
    len_a, len_b = len(seq_a), len(seq_b)
    score = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize first row and column with gap penalties
    for i in range(1, len_a + 1):
        score[i][0] = score[i-1][0] + gap_open + (i-1)*gap_extend
    for j in range(1, len_b + 1):
        score[0][j] = score[0][j-1] + gap_open + (j-1)*gap_extend
    
    # Fill the score matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate the score for matching/mismatch
            match_score = score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            
            # Calculate the score for gap in seq_a
            gap_a_score = score[i-1][j] + gap_extend
            
            # Calculate the score for gap in seq_b
            gap_b_score = score[i][j-1] + gap_extend
            
            # Choose the maximum score
            if mode == "global":
                score[i][j] = max(match_score, gap_a_score, gap_b_score)
            else:  # local mode
                score[i][j] = max(match_score, gap_a_score, gap_b_score, 0)
    
    # Backtrack to find the optimal alignment
    if mode == "global":
        i, j = len_a, len_b
    else:  # local mode
        # Find the maximum score in the matrix
        max_score = np.max(score)
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(score), score.shape)
        if max_score == 0:
            return (0, "", "")
    
    # Backtrack to reconstruct the alignment
    aligned_a = []
    aligned_b = []
    while i > 0 and j > 0:
        if mode == "global":
            # Check if the current cell is a match/mismatch
            if i > 0 and j > 0 and score[i][j] == score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif i > 0 and score[i][j] == score[i-1][j] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            else:  # j > 0 and score[i][j] == score[i][j-1] + gap_extend
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
        else:  # local mode
            # Check if the current cell is a match/mismatch
            if i > 0 and j > 0 and score[i][j] == score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif i > 0 and score[i][j] == score[i-1][j] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            else:  # j > 0 and score[i][j] == score[i][j-1] + gap_extend
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
            # If the score is 0, we've reached the end of the local alignment
            if score[i][j] == 0:
                break
    
    # Reverse the aligned sequences
    aligned_a = ''.join(reversed(aligned_a))
    aligned_b = ''.join(reversed(aligned_b))
    
    # Calculate the score
    score_value = score[i][j] if mode == "global" else max_score
    
    return (score_value, aligned_a, aligned_b)