import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    m, n = len(seq_a), len(seq_b)
    
    # Initialize DP tables
    A = [[0]*(n+1) for _ in range(m+1)]
    B = [[0]*(n+1) for _ in range(m+1)]
    C = [[0]*(n+1) for _ in range(m+1)]
    
    # Global mode initialization
    if mode == "global":
        for j in range(1, n+1):
            B[0][j] = gap_open + (j-1)*gap_extend
        for i in range(1, m+1):
            C[i][0] = gap_open + (i-1)*gap_extend
        
        # Fill DP tables
        for i in range(1, m+1):
            for j in range(1, n+1):
                a_char, b_char = seq_a[i-1], seq_b[j-1]
                score = match if a_char == b_char else mismatch
                
                # Compute A[i][j]
                A[i][j] = max(
                    A[i-1][j-1] + score,
                    B[i-1][j] + score,
                    C[i-1][j] + score
                )
                
                # Compute B[i][j]
                B[i][j] = max(
                    B[i-1][j] + gap_extend,
                    C[i-1][j] + gap_open,
                    A[i-1][j] + gap_open
                )
                
                # Compute C[i][j]
                C[i][j] = max(
                    C[i][j-1] + gap_extend,
                    B[i][j-1] + gap_open,
                    A[i][j-1] + gap_open
                )
        
        # Backtrack for global mode
        score = A[m][n]
        aligned_a, aligned_b = [], []
        i, j = m, n
        while i > 0 and j > 0:
            if A[i][j] == A[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif A[i][j] == B[i-1][j] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            elif A[i][j] == C[i-1][j] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
            else:
                break
        
        # Add remaining characters
        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        while j > 0:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
        
        aligned_a.reverse()
        aligned_b.reverse()
        
        return (score, ''.join(aligned_a), ''.join(aligned_b))
    
    elif mode == "local":
        # Local mode initialization
        for i in range(m+1):
            for j in range(n+1):
                # Initialize A[i][j]
                if i == 0 and j == 0:
                    A[i][j] = 0
                elif i == 0:
                    A[i][j] = 0  # No match possible
                elif j == 0:
                    A[i][j] = 0  # No match possible
                else:
                    a_char, b_char = seq_a[i-1], seq_b[j-1]
                    score = match if a_char == b_char else mismatch
                    A[i][j] = max(0, A[i-1][j-1] + score)
                
                # Initialize B[i][j]
                if i == 0 and j == 0:
                    B[i][j] = 0
                elif i == 0:
                    B[i][j] = gap_open + (j-1)*gap_extend
                elif j == 0:
                    B[i][j] = max(0, B[i-1][j] + gap_extend, A[i-1][j] + gap_open)
                else:
                    B[i][j] = max(0, B[i-1][j] + gap_extend, C[i-1][j] + gap_open, A[i-1][j] + gap_open)
                
                # Initialize C[i][j]
                if i == 0 and j == 0:
                    C[i][j] = 0
                elif i == 0:
                    C[i][j] = gap_open + (j-1)*gap_extend
                elif j == 0:
                    C[i][j] = max(0, C[i-1][j] + gap_extend, A[i-1][j] + gap_open)
                else:
                    C[i][j] = max(0, C[i][j-1] + gap_extend, B[i][j-1] + gap_open, A[i-1][j] + gap_open)
        
        # Find maximum score
        max_score = 0
        max_i, max_j = 0, 0
        for i in range(m+1):
            for j in range(n+1):
                if A[i][j] > max_score:
                    max_score = A[i][j]
                    max_i, max_j = i, j
                elif B[i][j] > max_score:
                    max_score = B[i][j]
                    max_i, max_j = i, j
                elif C[i][j] > max_score:
                    max_score = C[i][j]
                    max_i, max_j = i, j
        
        if max_score == 0:
            return (0, "", "")
        
        # Backtrack for local mode
        aligned_a, aligned_b = [], []
        i, j = max_i, max_j
        while i > 0 and j > 0:
            # Check which table the current value came from
            if A[i][j] == max(0, A[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif B[i][j] == max(0, B[i-1][j] + gap_extend, C[i-1][j] + gap_open, A[i-1][j] + gap_open):
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            elif C[i][j] == max(0, C[i][j-1] + gap_extend, B[i][j-1] + gap_open, A[i-1][j] + gap_open):
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
            else:
                break
        
        # Add remaining characters
        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        while j > 0:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
        
        aligned_a.reverse()
        aligned_b.reverse()
        
        return (max_score, ''.join(aligned_a), ''.join(aligned_b))

