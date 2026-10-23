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
    
    # Initialize first row and column
    for i in range(1, len_a + 1):
        score[i][0] = score[i-1][0] + gap_open + (i-1)*gap_extend
    for j in range(1, len_b + 1):
        score[0][j] = score[0][j-1] + gap_open + (j-1)*gap_extend
    
    # Fill the score matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate match/mismatch score
            if seq_a[i-1] == seq_b[j-1]:
                match_score = match
            else:
                match_score = mismatch
            
            # Calculate the three possible scores
            diag = score[i-1][j-1] + match_score
            up = score[i-1][j] + gap_extend
            left = score[i][j-1] + gap_extend
            
            # Choose the maximum score
            if mode == "global":
                score[i][j] = max(diag, up, left)
            else:  # local alignment
                score[i][j] = max(diag, up, left, 0)
    
    # Trace back to find the optimal alignment
    i, j = len_a, len_b
    aligned_a, aligned_b = ",", ""
    
    # Trace back for global alignment
    if mode == "global":
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif score[i][j] == score[i-1][j] + gap_extend:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
            elif score[i][j] == score[i][j-1] + gap_extend:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                j -= 1
            else:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
        
        # Handle remaining characters
        while i > 0:
            aligned_a = seq_a[i-1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        while j > 0:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j-1] + aligned_b
            j -= 1
    else:  # local alignment
        # Find the maximum score in the matrix
        max_score = np.max(score)
        if max_score <= 0:
            return (0, "", "")
        
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(score), score.shape)
        
        # Trace back from the maximum score
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            elif score[i][j] == score[i-1][j] + gap_extend:
                aligned_a = "-" + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
            elif score[i][j] == score[i][j-1] + gap_extend:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = "-" + aligned_b
                j -= 1
            else:
                aligned_a = seq_a[i-1] + aligned_a
                aligned_b = seq_b[j-1] + aligned_b
                i -= 1
                j -= 1
            
            # If the score becomes zero, we stop tracing
            if score[i][j] == 0:
                break
    
    # Check for double-gap columns and adjust alignment if needed
    # This is a simplified check to prevent double gaps
    for k in range(len(aligned_a) - 1):
        if aligned_a[k] == "-" and aligned_b[k] == "-":
            # Adjust the alignment by inserting a gap in the other sequence
            # This is a simplified approach and may need more complex logic
            aligned_a = aligned_a[:k] + "-" + aligned_a[k+1:]
            aligned_b = aligned_b[:k] + seq_a[k] + aligned_b[k+1:]
            # This is a simplified fix and may not handle all cases
            # A more robust solution would require a different approach
    
    # Return the score and aligned sequences
    return (score[len_a][len_b], aligned_a, aligned_b)