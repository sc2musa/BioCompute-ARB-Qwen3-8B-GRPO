import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    n, m = len(seq_a), len(seq_b)
    
    # Initialize score matrices
    M = np.full((n + 1, m + 1), -np.inf, dtype=int)
    X = np.full((n + 1, m + 1), -np.inf, dtype=int)
    Y = np.full((n + 1, m + 1), -np.inf, dtype=int)
    
    # Initialize borders
    if mode == "global":
        M[0][0] = 0
        for i in range(1, n + 1):
            X[i][0] = gap_open + (i - 1) * gap_extend
        for j in range(1, m + 1):
            Y[0][j] = gap_open + (j - 1) * gap_extend
    elif mode == "local":
        for i in range(n + 1):
            for j in range(m + 1):
                M[i][j] = max(0, M[i][j])
    
    # Fill the matrices
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Calculate the score for M[i][j]
            score = match if seq_a[i - 1] == seq_b[j - 1] else mismatch
            M[i][j] = score + max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1])
            
            # Calculate the score for X[i][j]
            X[i][j] = max(
                M[i - 1][j] + gap_open, 
                X[i - 1][j] + gap_extend, 
                Y[i - 1][j] + gap_open
            )
            
            # Calculate the score for Y[i][j]
            Y[i][j] = max(
                M[i][j - 1] + gap_open, 
                Y[i][j - 1] + gap_extend, 
                X[i][j - 1] + gap_open
            )
    
    # Determine the best score and end cell
    if mode == "global":
        best_score = max(M[n][m], X[n][m], Y[n][m])
        end_cell = np.unravel_index([M[n][m], X[n][m], Y[n][m]].argmax(), (3, n + 1, m + 1))
    elif mode == "local":
        best_score = M.max()
        end_cell = np.unravel_index(M.argmax(), (n + 1, m + 1))
    
    # Backtrack to find the alignment
    aligned_a = []
    aligned_b = []
    i, j = end_cell[1], end_cell[2]
    
    # Backtrack from the end cell
    while i > 0 and j > 0:
        if mode == "global":
            if M[i][j] == max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1]):
                aligned_a.append(seq_a[i - 1])
                aligned_b.append(seq_b[j - 1])
                i -= 1
                j -= 1
            elif X[i][j] == max(M[i - 1][j] + gap_open, X[i - 1][j] + gap_extend, Y[i - 1][j] + gap_open):
                aligned_a.append(seq_a[i - 1])
                aligned_b.append('-')
                i -= 1
            elif Y[i][j] == max(M[i][j - 1] + gap_open, Y[i][j - 1] + gap_extend, X[i][j - 1] + gap_open):
                aligned_a.append('-')
                aligned_b.append(seq_b[j - 1])
                j -= 1
        elif mode == "local":
            if M[i][j] == max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1], 0):
                aligned_a.append(seq_a[i - 1])
                aligned_b.append(seq_b[j - 1])
                i -= 1
                j -= 1
            elif X[i][j] == max(M[i - 1][j] + gap_open, X[i - 1][j] + gap_extend, Y[i - 1][j] + gap_open, 0):
                aligned_a.append(seq_a[i - 1])
                aligned_b.append('-')
                i -= 1
            elif Y[i][j] == max(M[i][j - 1] + gap_open, Y[i][j - 1] + gap_extend, X[i][j - 1] + gap_open, 0):
                aligned_a.append('-')
                aligned_b.append(seq_b[j - 1])
                j -= 1
    
    # Handle the remaining characters
    while i > 0:
        aligned_a.append(seq_a[i - 1])
        aligned_b.append('-')
        i -= 1
    while j > 0:
        aligned_a.append('-')
        aligned_b.append(seq_b[j - 1])
        j -= 1
    
    # Reverse the alignments to get the correct order
    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]
    
    # Calculate the score
    score = 0
    for a, b in zip(aligned_a, aligned_b):
        if a == '-' and b == '-':
            continue
        elif a == '-':
            score += gap_extend if aligned_b.count('-') > 1 else gap_open
        elif b == '-':
            score += gap_extend if aligned_a.count('-') > 1 else gap_open
        else:
            score += match if a == b else mismatch
    
    return (score, ''.join(aligned_a), ''.join(aligned_b))
