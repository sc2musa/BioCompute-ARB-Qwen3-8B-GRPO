import numpy as np

def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Handle empty sequences
    if not seq_a:
        return (0, seq_b, "")
    if not seq_b:
        return (0, seq_a, "")
    
    # Initialize scoring matrix
    len_a, len_b = len(seq_a), len(seq_b)
    score = np.zeros((len_a+1, len_b+1), dtype=int)
    
    # Initialize first row and column with gap penalties
    for i in range(1, len_a+1):
        score[i, 0] = score[i-1, 0] + gap_open + (i-1)*gap_extend
    for j in range(1, len_b+1):
        score[0, j] = score[0, j-1] + gap_open + (j-1)*gap_extend
    
    # Fill the matrix
    for i in range(1, len_a+1):
        for j in range(1, len_b+1):
            # Calculate scores for match/mismatch and gaps
            match_score = score[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            gap_a = score[i-1, j] + gap_open
            gap_b = score[i, j-1] + gap_open
            
            # Choose the maximum score
            score[i, j] = max(match_score, gap_a, gap_b)
    
    # Find the optimal score
    if mode == "global":
        score_value = score[len_a, len_b]
    else:
        # For local mode, find the maximum value in the matrix
        score_value = np.max(score)
        # Find the position of the maximum value
        i, j = np.unravel_index(np.argmax(score), score.shape)
        
    # Backtrack to find the alignment
    aligned_a = ""
    aligned_b = ""
    i, j = len_a, len_b
    
    while i > 0 and j > 0:
        if score[i, j] == score[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            # Match/mismatch
            aligned_a += seq_a[i-1]
            aligned_b += seq_b[j-1]
            i -= 1
            j -= 1
        elif score[i, j] == score[i-1, j] + gap_open:
            # Gap in seq_a
            aligned_a += "-"
            aligned_b += seq_b[j-1]
            i -= 1
        else:
            # Gap in seq_b
            aligned_a += seq_a[i-1]
            aligned_b += "-"
            j -= 1
    
    # Handle remaining characters
    while i > 0:
        aligned_a += seq_a[i-1]
        aligned_b += "-"
        i -= 1
    while j > 0:
        aligned_a += "-"
        aligned_b += seq_b[j-1]
        j -= 1
    
    # Reverse the alignments
    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]
    
    return (score_value, aligned_a, aligned_b)