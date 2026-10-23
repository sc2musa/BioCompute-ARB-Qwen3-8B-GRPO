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
        for i in range(1, len_a + 1):
            score[i][0] = score[i-1][0] + gap_open
        for j in range(1, len_b + 1):
            score[0][j] = score[0][j-1] + gap_open
    elif mode == "local":
        for i in range(1, len_a + 1):
            score[i][0] = 0
        for j in range(1, len_b + 1):
            score[0][j] = 0
    
    # Fill the score matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Match/mismatch score
            match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            
            # Gap in seq_a
            gap_a = score[i-1][j] + gap_extend
            
            # Gap in seq_b
            gap_b = score[i][j-1] + gap_extend
            
            # Diagonal score
            diag_score = score[i-1][j-1] + match_score
            
            # Choose the maximum score
            score[i][j] = max(diag_score, gap_a, gap_b)
            
    # Determine the optimal score based on mode
    if mode == "global":
        optimal_score = score[len_a][len_b]
    elif mode == "local":
        optimal_score = np.max(score)
        # Find the position of the optimal score
        for i in range(len_a, 0, -1):
            for j in range(len_b, 0, -1):
                if score[i][j] == optimal_score:
                    # Backtrack to find the alignment
                    aligned_a = []
                    aligned_b = []
                    while i > 0 and j > 0:
                        if score[i][j] == score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
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
                    # Reverse the alignment to get the correct order
                    aligned_a = aligned_a[::-1]
                    aligned_b = aligned_b[::-1]
                    # Trim the alignment to the optimal score
                    # (This part needs to be implemented)
                    break
    
    # Return the score and aligned sequences
    return (optimal_score, "".join(aligned_a), "".join(aligned_b))