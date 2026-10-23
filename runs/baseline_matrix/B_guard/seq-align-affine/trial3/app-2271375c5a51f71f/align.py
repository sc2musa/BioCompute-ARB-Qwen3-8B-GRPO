import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    len_a, len_b = len(seq_a), len(seq_b)
    
    # Initialize the scoring matrix
    score_matrix = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize the first row and column for global mode
    if mode == "global":
        for i in range(1, len_a + 1):
            score_matrix[i][0] = score_matrix[i-1][0] + gap_open + gap_extend
        for j in range(1, len_b + 1):
            score_matrix[0][j] = score_matrix[0][j-1] + gap_open + gap_extend
    
    # Initialize the scoring matrix for local mode
    elif mode == "local":
        # Initialize with -infinity to ensure only positive scores are considered
        score_matrix = np.full((len_a + 1, len_b + 1), -np.inf, dtype=int)
        score_matrix[0][0] = 0
        
        # Initialize the first row and column for local mode
        for i in range(1, len_a + 1):
            score_matrix[i][0] = max(score_matrix[i-1][0] + gap_extend, 0)
        for j in range(1, len_b + 1):
            score_matrix[0][j] = max(score_matrix[0][j-1] + gap_extend, 0)
    
    # Fill the scoring matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate the score for match/mismatch
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Calculate the score for gap in seq_a
            gap_a_score = score_matrix[i-1][j] + gap_extend
            
            # Calculate the score for gap in seq_b
            gap_b_score = score_matrix[i][j-1] + gap_extend
            
            # Calculate the score for match/mismatch
            score = max(match_score, gap_a_score, gap_b_score)
            
            # Update the score matrix
            if mode == "global":
                score_matrix[i][j] = score
            elif mode == "local":
                score_matrix[i][j] = max(score, 0)
    
    # Determine the maximum score based on the mode
    if mode == "global":
        max_score = score_matrix[len_a][len_b]
    elif mode == "local":
        max_score = np.max(score_matrix)
    else:
        raise ValueError("Invalid mode. Use 'global' or 'local'.")
    
    # Traceback to find the optimal alignment
    aligned_a = ""
    aligned_b = ""
    i, j = len_a, len_b
    
    # Find the position of the maximum score for local mode
    if mode == "local":
        max_score_pos = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        i, j = max_score_pos
    
    # Traceback to find the optimal alignment
    while i > 0 and j > 0:
        if mode == "global":
            if score_matrix[i][j] == score_matrix[i-1][j] + gap_extend:
                # Gap in seq_a
                aligned_a += "-"
                aligned_b += seq_b[j-1]
                i -= 1
            elif score_matrix[i][j] == score_matrix[i][j-1] + gap_extend:
                # Gap in seq_b
                aligned_a += seq_a[i-1]
                aligned_b += "-"
                j -= 1
            else:
                # Match/mismatch
                aligned_a += seq_a[i-1]
                aligned_b += seq_b[j-1]
                i -= 1
                j -= 1
        elif mode == "local":
            if score_matrix[i][j] == score_matrix[i-1][j] + gap_extend:
                # Gap in seq_a
                aligned_a += "-"
                aligned_b += seq_b[j-1]
                i -= 1
            elif score_matrix[i][j] == score_matrix[i][j-1] + gap_extend:
                # Gap in seq_b
                aligned_a += seq_a[i-1]
                aligned_b += "-"
                j -= 1
            else:
                # Match/mismatch
                aligned_a += seq_a[i-1]
                aligned_b += seq_b[j-1]
                i -= 1
                j -= 1
    
    # Reverse the alignment to get the correct order
    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]
    
    # Check if the alignment is valid for local mode
    if mode == "local" and max_score == 0:
        return (0, "", "")
    
    # Return the score and the aligned sequences
    return (max_score, aligned_a, aligned_b)