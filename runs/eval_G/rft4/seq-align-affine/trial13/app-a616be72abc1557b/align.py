import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)

    # Handle empty sequences
    if not seq_a and not seq_b:
        return (0, "", "")
    if not seq_a:
        if mode == "global":
            score = gap_open + (len_b - 1) * gap_extend
            return (score, "-" * len_b, seq_b)
        else:
            score = max(gap_open + (len_b - 1) * gap_extend, 0)
            return (score, "-" * len_b, seq_b) if score > 0 else (0, "", "")
    if not seq_b:
        if mode == "global":
            score = gap_open + (len_a - 1) * gap_extend
            return (score, seq_a, "-" * len_a)
        else:
            score = max(gap_open + (len_a - 1) * gap_extend, 0)
            return (score, seq_a, "-" * len_a) if score > 0 else (0, "", "")

    # Initialize DP tables
    M = [[0]*(len_b+1) for _ in range(len_a+1)]
    X = [[-float('inf')]*(len_b+1) for _ in range(len_a+1)]
    Y = [[-float('inf')]*(len_b+1) for _ in range(len_a+1)]

    # Global mode initialization
    if mode == "global":
        for j in range(1, len_b+1):
            X[0][j] = gap_open + (j-1)*gap_extend
        for i in range(1, len_a+1):
            Y[i][0] = gap_open + (i-1)*gap_extend
    else:
        # Local mode initialization
        for i in range(len_a+1):
            for j in range(len_b+1):
                M[i][j] = -float('inf')
        M[0][0] = 0

    # Fill DP tables
    for i in range(1, len_a+1):
        for j in range(1, len_b+1):
            # Compute M[i][j]
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            M[i][j] = max(
                M[i-1][j-1] + score,
                X[i-1][j] + score,
                Y[i-1][j] + score,
                0 if mode == "local" else -float('inf')
            )

            # Compute X[i][j]
            if mode == "local":
                X[i][j] = max(
                    X[i-1][j] + gap_extend,
                    Y[i][j-1] + gap_open,
                    M[i-1][j-1] + gap_open,
                    0
                )
            else:
                X[i][j] = max(
                    X[i-1][j] + gap_extend,
                    Y[i][j-1] + gap_open,
                    M[i-1][j-1] + gap_open
                )

            # Compute Y[i][j]
            if mode == "local":
                Y[i][j] = max(
                    Y[i-1][j] + gap_extend,
                    X[i][j-1] + gap_open,
                    M[i-1][j-1] + gap_open,
                    0
                )
            else:
                Y[i][j] = max(
                    Y[i-1][j] + gap_extend,
                    X[i][j-1] + gap_open,
                    M[i-1][j-1] + gap_open
                )

    # Determine best score
    if mode == "global":
        best_score = M[len_a][len_b]
    else:
        # Find the maximum value across all cells
        best_score = -float('inf')
        for i in range(len_a+1):
            for j in range(len_b+1):
                best_score = max(best_score, M[i][j], X[i][j], Y[i][j])
        if best_score < 0:
            return (0, "", "")

    # Backtrack to find alignment
    def backtrack(i, j):
        path = []
        while i > 0 and j > 0:
            # Determine which state we came from
            if mode == "global":
                current = M[i][j]
            else:
                current = max(M[i][j], X[i][j], Y[i][j])
            
            # Check M[i][j]
            if current == M[i][j]:
                path.append((seq_a[i-1], seq_b[j-1]))
                i -= 1
                j -= 1
            elif current == X[i][j]:
                # Check which option contributed to X[i][j]
                if X[i][j] == X[i-1][j] + gap_extend:
                    path.append(('-', seq_b[j-1]))
                    i -= 1
                elif X[i][j] == Y[i][j-1] + gap_open:
                    path.append(('-', seq_b[j-1]))
                    j -= 1
                elif X[i][j] == M[i-1][j-1] + gap_open:
                    path.append(('-', seq_b[j-1]))
                    i -= 1
                else:
                    path.append(('-', seq_b[j-1]))
                    i -= 1
            elif current == Y[i][j]:
                # Check which option contributed to Y[i][j]
                if Y[i][j] == Y[i-1][j] + gap_extend:
                    path.append((seq_a[i-1], '-'))
                    j -= 1
                elif Y[i][j] == X[i][j-1] + gap_open:
                    path.append((seq_a[i-1], '-'))
                    i -= 1
                elif Y[i][j] == M[i-1][j-1] + gap_open:
                    path.append((seq_a[i-1], '-'))
                    j -= 1
                else:
                    path.append((seq_a[i-1], '-'))
                    j -= 1
        # Handle remaining i or j
        while i > 0:
            path.append((seq_a[i-1], '-'))
            i -= 1
        while j > 0:
            path.append(('-', seq_b[j-1]))
            j -= 1
        return path[::-1]

    # Reconstruct alignment
    if mode == "global":
        path = backtrack(len_a, len_b)
    else:
        # Find the cell with the best score
        best_i, best_j = 0, 0
        for i in range(len_a+1):
            for j in range(len_b+1):
                if (mode == "global" and M[i][j] == best_score) or (mode == "local" and max(M[i][j], X[i][j], Y[i][j]) == best_score):
                    best_i, best_j = i, j
        path = backtrack(best_i, best_j)

    # Build aligned sequences
    aligned_a, aligned_b = "", ""
    for a, b in path:
        aligned_a += a
        aligned_b += b

    return (best_score, aligned_a, aligned_b)