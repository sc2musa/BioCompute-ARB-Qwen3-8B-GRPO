import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    n, m = len(seq_a), len(seq_b)
    
    # Initialize matrices
    M = np.full((n+1, m+1), -np.inf, dtype=int)
    X = np.full((n+1, m+1), -np.inf, dtype=int)
    Y = np.full((n+1, m+1), -np.inf, dtype=int)
    
    # Backpointers
    back_M = np.full((n+1, m+1), ("", 0, 0), dtype=object)
    back_X = np.full((n+1, m+1), ("", 0, 0), dtype=object)
    back_Y = np.full((n+1, m+1), ("", 0, 0), dtype=object)
    
    # Global mode initial conditions
    M[0][0] = 0
    for i in range(1, n+1):
        X[i][0] = gap_open + (i-1)*gap_extend
    for j in range(1, m+1):
        Y[0][j] = gap_open + (j-1)*gap_extend
    
    # Fill matrices
    for i in range(1, n+1):
        for j in range(1, m+1):
            # Compute M[i][j]
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            if mode == "global":
                prev_max = max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1])
            else:
                prev_max = max(0, M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1])
            M[i][j] = score + prev_max
            
            # Determine backpointer for M[i][j]
            if mode == "global":
                options = [M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1]]
            else:
                options = [M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1], 0]
            prev_max_val = max(options)
            
            if prev_max_val == M[i-1][j-1]:
                back_M[i][j] = ("M", i-1, j-1)
            elif prev_max_val == X[i-1][j-1]:
                back_M[i][j] = ("X", i-1, j-1)
            elif prev_max_val == Y[i-1][j-1]:
                back_M[i][j] = ("Y", i-1, j-1)
            elif prev_max_val == 0:
                back_M[i][j] = ("start", 0, 0)
            
            # Compute X[i][j]
            option1 = M[i-1][j] + gap_open
            option2 = X[i-1][j] + gap_extend
            option3 = Y[i-1][j] + gap_open
            prev_max_x = max(option1, option2, option3)
            X[i][j] = prev_max_x
            
            # Determine backpointer for X[i][j]
            if option1 == prev_max_x:
                back_X[i][j] = ("M", i-1, j)
            elif option2 == prev_max_x:
                back_X[i][j] = ("X", i-1, j)
            elif option3 == prev_max_x:
                back_X[i][j] = ("Y", i-1, j)
            
            # Compute Y[i][j]
            option1 = M[i][j-1] + gap_open
            option2 = Y[i][j-1] + gap_extend
            option3 = X[i][j-1] + gap_open
            prev_max_y = max(option1, option2, option3)
            Y[i][j] = prev_max_y
            
            # Determine backpointer for Y[i][j]
            if option1 == prev_max_y:
                back_Y[i][j] = ("M", i, j-1)
            elif option2 == prev_max_y:
                back_Y[i][j] = ("Y", i, j-1)
            elif option3 == prev_max_y:
                back_Y[i][j] = ("X", i, j-1)
    
    # Determine best score and cell
    if mode == "global":
        best_score = max(M[n][m], X[n][m], Y[n][m])
        best_cells = [(n, m, "M"), (n, m, "X"), (n, m, "Y")]
        for i, j, matrix in best_cells:
            if eval(f"{matrix}[n][m]") == best_score:
                best_i, best_j, best_matrix = i, j, matrix
                break
    else:
        best_score = max(M.max(), 0)
        best_i, best_j, best_matrix = 0, 0, "M"
        for i in range(n+1):
            for j in range(m+1):
                if M[i][j] == best_score:
                    best_i, best_j, best_matrix = i, j, "M"
                    break
            else:
                continue
            break
        if best_score == 0:
            return (0, "", "")
    
    # Backtrack to get aligned sequences
    aligned_a = []
    aligned_b = []
    current_i, current_j, current_matrix = best_i, best_j, best_matrix
    
    while current_i != 0 or current_j != 0:
        if current_matrix == "M":
            aligned_a.append(seq_a[current_i-1])
            aligned_b.append(seq_b[current_j-1])
            prev_matrix, prev_i, prev_j = back_M[current_i][current_j]
        elif current_matrix == "X":
            aligned_a.append(seq_a[current_i-1])
            aligned_b.append("-")
            prev_matrix, prev_i, prev_j = back_X[current_i][current_j]
        elif current_matrix == "Y":
            aligned_a.append("-")
            aligned_b.append(seq_b[current_j-1])
            prev_matrix, prev_i, prev_j = back_Y[current_i][current_j]
        elif current_matrix == "start":
            aligned_a.append(seq_a[current_i-1])
            aligned_b.append(seq_b[current_j-1])
            prev_matrix, prev_i, prev_j = back_M[current_i][current_j]
        
        current_i, current_j, current_matrix = prev_i, prev_j, prev_matrix
    
    aligned_a.reverse()
    aligned_b.reverse()
    
    return (best_score, "".join(aligned_a), "".join(aligned_b))
