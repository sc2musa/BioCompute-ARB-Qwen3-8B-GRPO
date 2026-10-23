import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)
    if len_a == 0 and len_b == 0:
        return (0, "", "")

    # Initialize matrices
    n, m = len_a, len_b
    M = np.full((n+1, m+1), -np.inf, dtype=int)
    X = np.full((n+1, m+1), -np.inf, dtype=int)
    Y = np.full((n+1, m+1), -np.inf, dtype=int)
    ptr = np.full((n+1, m+1), "", dtype=object)

    # Global mode initialization
    if mode == "global":
        M[0][0] = 0
        for i in range(1, n+1):
            X[i][0] = gap_open + (i-1)*gap_extend
        for j in range(1, m+1):
            Y[0][j] = gap_open + (j-1)*gap_extend
    else:  # Local mode
        M[0][0] = 0

    # Fill matrices
    for i in range(1, n+1):
        for j in range(1, m+1):
            # Calculate score for current column
            a_char = seq_a[i-1]
            b_char = seq_b[j-1]
            score = match if a_char == b_char else mismatch

            # Calculate M[i][j]
            if mode == "global":
                prev_max = max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1])
            else:
                prev_max = max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1], 0)
            M[i][j] = score + prev_max
            # Determine source for M[i][j]
            prev_vals = [M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1]]
            max_val = max(prev_vals)
            if prev_vals[0] == max_val:
                ptr[i][j] = "M"
            elif prev_vals[1] == max_val:
                ptr[i][j] = "X"
            else:
                ptr[i][j] = "Y"

            # Calculate X[i][j]
            option1 = M[i-1][j] + gap_open
            option2 = X[i-1][j] + gap_extend
            option3 = Y[i-1][j] + gap_open
            X[i][j] = max(option1, option2, option3)
            # Determine source for X[i][j]
            prev_vals = [M[i-1][j], X[i-1][j], Y[i-1][j]]
            max_val = max(prev_vals)
            if prev_vals[0] == max_val:
                ptr[i][j] = "M"
            elif prev_vals[1] == max_val:
                ptr[i][j] = "X"
            else:
                ptr[i][j] = "Y"

            # Calculate Y[i][j]
            option1 = M[i][j-1] + gap_open
            option2 = Y[i][j-1] + gap_extend
            option3 = X[i][j-1] + gap_open
            Y[i][j] = max(option1, option2, option3)
            # Determine source for Y[i][j]
            prev_vals = [M[i][j-1], Y[i][j-1], X[i][j-1]]
            max_val = max(prev_vals)
            if prev_vals[0] == max_val:
                ptr[i][j] = "M"
            elif prev_vals[1] == max_val:
                ptr[i][j] = "Y"
            else:
                ptr[i][j] = "X"

    # Determine best score and end cell
    if mode == "global":
        best_score = max(M[n][m], X[n][m], Y[n][m])
        end_i, end_j = (n, m) if M[n][m] == best_score else (n, m) if X[n][m] == best_score else (n, m)
    else:  # Local mode
        best_score = np.max(M)
        if best_score < 0:
            return (0, "", "")
        # Find the cell with the maximum score
        for i in range(n+1):
            for j in range(m+1):
                if M[i][j] == best_score:
                    end_i, end_j = i, j
                    break
            else:
                continue
            break

    # Backtrack to find alignment
    aligned_a = []
    aligned_b = []
    i, j = end_i, end_j
    while i > 0 and j > 0:
        if ptr[i][j] == "M":
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif ptr[i][j] == "X":
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
            j -= 0
        else:  # Y
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            i -= 0
            j -= 1

    # Reverse to get correct order
    aligned_a = aligned_a[::-1]
    aligned_b = aligned_b[::-1]

    # Handle edge cases (e.g., when one sequence is empty)
    if mode == "global":
        # Ensure both sequences are fully aligned
        while len(aligned_a) < len_a:
            aligned_a.append("-")
        while len(aligned_b) < len_b:
            aligned_b.append("-")
    else:
        # Ensure no leading or trailing gaps
        while aligned_a and aligned_a[0] == "-":
            aligned_a.pop(0)
            aligned_b.pop(0)
        while aligned_a and aligned_a[-1] == "-":
            aligned_a.pop()
            aligned_b.pop()

    return (best_score, "".join(aligned_a), "".join(aligned_b))