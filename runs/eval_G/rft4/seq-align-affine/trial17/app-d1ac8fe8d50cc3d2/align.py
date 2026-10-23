import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    if not seq_a and not seq_b:
        return (0, "", "")
    elif not seq_a:
        return (gap_open + (len(seq_b) - 1)*gap_extend, "-"*len(seq_b), seq_b)
    elif not seq_b:
        return (gap_open + (len(seq_a) - 1)*gap_extend, seq_a, "-"*len(seq_a))

    n, m = len(seq_a), len(seq_b)
    M = np.full((n+1, m+1), -np.inf)
    X = np.full((n+1, m+1), -np.inf)
    Y = np.full((n+1, m+1), -np.inf)
    back_M = np.full((n+1, m+1), "")
    back_X = np.full((n+1, m+1), "")
    back_Y = np.full((n+1, m+1), "")

    if mode == "global":
        M[0][0] = 0
        for j in range(1, m+1):
            X[0][j] = gap_open + (j-1)*gap_extend
        for i in range(1, n+1):
            Y[i][0] = gap_open + (i-1)*gap_extend
    else:
        M[0][0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            # Compute M[i][j]
            option1 = M[i-1][j-1] + score
            option2 = X[i-1][j] + score
            option3 = Y[i-1][j] + score
            max_val = max(option1, option2, option3)
            M[i][j] = max_val
            if option1 == max_val:
                back_M[i][j] = 'M'
            elif option2 == max_val:
                back_M[i][j] = 'X'
            else:
                back_M[i][j] = 'Y'
            
            # Compute X[i][j]
            option1_x = X[i-1][j] + gap_extend
            option2_x = Y[i-1][j] + gap_open
            option3_x = M[i-1][j] + gap_open
            max_val_x = max(option1_x, option2_x, option3_x)
            X[i][j] = max_val_x
            if option1_x == max_val_x:
                back_X[i][j] = 'X'
            elif option2_x == max_val_x:
                back_X[i][j] = 'Y'
            else:
                back_X[i][j] = 'M'
            
            # Compute Y[i][j]
            option1_y = Y[i-1][j] + gap_extend
            option2_y = X[i-1][j] + gap_open
            option3_y = M[i-1][j] + gap_open
            max_val_y = max(option1_y, option2_y, option3_y)
            Y[i][j] = max_val_y
            if option1_y == max_val_y:
                back_Y[i][j] = 'Y'
            elif option2_y == max_val_y:
                back_Y[i][j] = 'X'
            else:
                back_Y[i][j] = 'M'

    if mode == "global":
        score = M[n][m]
        i, j = n, m
        aligned_a = []
        aligned_b = []
        while i > 0 or j > 0:
            if back_M[i][j] == 'M':
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif back_M[i][j] == 'X':
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            elif back_M[i][j] == 'Y':
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
    else:
        max_val = -np.inf
        max_i, max_j = 0, 0
        for i in range(n+1):
            for j in range(m+1):
                current = max(M[i][j], X[i][j], Y[i][j])
                if current > max_val:
                    max_val = current
                    max_i, max_j = i, j
        if max_val <= 0:
            return (0, "", "")
        i, j = max_i, max_j
        aligned_a = []
        aligned_b = []
        while i > 0 or j > 0:
            if M[i][j] == max_val:
                current_state = 'M'
            elif X[i][j] == max_val:
                current_state = 'X'
            else:
                current_state = 'Y'
            if current_state == 'M':
                if i > 0 and j > 0:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                    max_val -= (match if seq_a[i-1] == seq_b[j-1] else mismatch)
                else:
                    break
            elif current_state == 'X':
                if i > 0:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    i -= 1
                    max_val -= gap_extend
                else:
                    break
            elif current_state == 'Y':
                if j > 0:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    j -= 1
                    max_val -= gap_extend
                else:
                    break
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
        score = max_val

    return (score, aligned_a, aligned_b)