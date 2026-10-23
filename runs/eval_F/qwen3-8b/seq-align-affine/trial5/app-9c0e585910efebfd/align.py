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
        # Needleman-Wunsch initialization
        for i in range(len_a + 1):
            score[i][0] = i * gap_open
        for j in range(len_b + 1):
            score[0][j] = j * gap_open
    elif mode == "local":
        # Smith-Waterman initialization
        for i in range(len_a + 1):
            score[i][0] = 0
        for j in range(len_b + 1):
            score[0][j] = 0
    else:
        raise ValueError("Invalid mode. Use 'global' or 'local'.")
    
    # Fill the score matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate the score for matching/mismatch
            match_score = score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            
            # Calculate the score for gap in seq_a
            gap_a_score = score[i-1][j] + gap_open
            
            # Calculate the score for gap in seq_b
            gap_b_score = score[i][j-1] + gap_open
            
            # Calculate the score for extending a gap in seq_a
            gap_a_extend_score = score[i-1][j] + gap_extend
            
            # Calculate the score for extending a gap in seq_b
            gap_b_extend_score = score[i][j-1] + gap_extend
            
            # Choose the maximum score
            score[i][j] = max(match_score, gap_a_score, gap_b_score, gap_a_extend_score, gap_b_extend_score)
    
    # Determine the optimal score based on the mode
    if mode == "global":
        score_val = score[len_a][len_b]
    elif mode == "local":
        # Find the maximum score in the matrix
        score_val = np.max(score)
        # Find the position of the maximum score
        max_i, max_j = np.unravel_index(np.argmax(score), score.shape)
    else:
        raise ValueError("Invalid mode. Use 'global' or 'local'.")
    
    # Backtrack to find the optimal alignment
    aligned_a = []
    aligned_b = []
    i, j = len_a, len_b
    
    if mode == "global":
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
            elif j > 0 and score[i][j] == score[i][j-1] + gap_open:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
            else:
                # Handle gap extension cases
                if i > 0 and score[i][j] == score[i-1][j] + gap_extend:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    i -= 1
                elif j > 0 and score[i][j] == score[i][j-1] + gap_extend:
                    aligned_a.append("-")
                    aligned_b.append(seq_b[j-1])
                    j -= 1
                else:
                    # This should not happen for global alignment
                    break
    elif mode == "local":
        # Backtrack from the maximum score position
        i, j = max_i, max_j
        while i > 0 and j > 0 and score[i][j] != 0:
            if i > 0 and j > 0 and score[i][j] == score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif i > 0 and score[i][j] == score[i-1][j] + gap_open:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            elif j > 0 and score[i][j] == score[i][j-1] + gap_open:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
            else:
                # Handle gap extension cases
                if i > 0 and score[i][j] == score[i-1][j] + gap_extend:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    i -= 1
                elif j > 0 and score[i][j] == score[i][j-1] + gap_extend:
                    aligned_a.append("-")
                    aligned_b.append(seq_b[j-1])
                    j -= 1
                else:
                    # This should not happen for local alignment
                    break
    
    # Reverse the alignment to get the correct order
    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]
    
    # Check if the alignment is valid
    if mode == "local" and score_val == 0:
        return (0, "", "")
    
    return (score_val, ''.join(aligned_a), ''.join(aligned_b))