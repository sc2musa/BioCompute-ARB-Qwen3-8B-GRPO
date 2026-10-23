import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a = len(seq_a)
    len_b = len(seq_b)

    if len_a == 0 and len_b == 0:
        return (0, "", "")

    # Initialize DP tables
    M = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
    A = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
    B = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]

    # Set initial conditions for global mode
    if mode == "global":
        # For i=0 and j=0
        M[0][0] = 0
        # For i=0 and j>0
        for j in range(1, len_b + 1):
            A[0][j] = gap_open + (j - 1) * gap_extend
        # For j=0 and i>0
        for i in range(1, len_a + 1):
            B[i][0] = gap_open + (i - 1) * gap_extend
    else:
        # For local mode, initialize all to 0
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                M[i][j] = 0
                A[i][j] = 0
                B[i][j] = 0

    # Fill DP tables
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Compute score for match/mismatch
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch

            # Compute M[i][j]
            m_val = max(M[i-1][j-1] + score, A[i-1][j-1] + score, B[i-1][j-1] + score)

            # Compute A[i][j]
            a_val = max(A[i-1][j] + gap_extend, max(M[i-1][j], B[i-1][j]) + gap_open)

            # Compute B[i][j]
            b_val = max(B[i][j-1] + gap_extend, max(M[i][j-1], A[i][j-1]) + gap_open)

            M[i][j] = m_val
            A[i][j] = a_val
            B[i][j] = b_val

    # Determine the optimal score
    if mode == "global":
        score = M[len_a][len_b]
        # Backtrack to find the alignment
        aligned_a, aligned_b = backtrack_global(M, A, B, seq_a, seq_b, len_a, len_b)
    else:
        # Find the maximum score in all cells
        max_score = -float('inf')
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if M[i][j] > max_score:
                    max_score = M[i][j]
                if A[i][j] > max_score:
                    max_score = A[i][j]
                if B[i][j] > max_score:
                    max_score = B[i][j]
        if max_score <= 0:
            return (0, "", "")
        # Find the cell with the maximum score
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if M[i][j] == max_score:
                    aligned_a, aligned_b = backtrack_local(M, A, B, seq_a, seq_b, i, j)
                    break
            else:
                continue
            break
        score = max_score

    return (score, aligned_a, aligned_b)


def backtrack_global(M, A, B, seq_a, seq_b, len_a, len_b):
    aligned_a = []
    aligned_b = []
    i, j = len_a, len_b
    while i > 0 and j > 0:
        if M[i][j] == M[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif A[i][j] == A[i-1][j] + gap_extend:
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            i -= 1
        elif B[i][j] == B[i][j-1] + gap_extend:
            aligned_a.append(seq_a[i-1])
            aligned_b.append('-')
            j -= 1
    # Handle remaining characters
    while i > 0:
        aligned_a.append(seq_a[i-1])
        aligned_b.append('-')
        i -= 1
    while j > 0:
        aligned_a.append('-')
        aligned_b.append(seq_b[j-1])
        j -= 1
    return ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b))

def backtrack_local(M, A, B, seq_a, seq_b, i, j):
    aligned_a = []
    aligned_b = []
    while i > 0 and j > 0:
        if M[i][j] == M[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif A[i][j] == A[i-1][j] + gap_extend:
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            i -= 1
        elif B[i][j] == B[i][j-1] + gap_extend:
            aligned_a.append(seq_a[i-1])
            aligned_b.append('-')
            j -= 1
        else:
            # This is a special case for local mode where the alignment ends
            break
    # Handle remaining characters
    while i > 0:
        aligned_a.append(seq_a[i-1])
        aligned_b.append('-')
        i -= 1
    while j > 0:
        aligned_a.append('-')
        aligned_b.append(seq_b[j-1])
        j -= 1
    return ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b))
