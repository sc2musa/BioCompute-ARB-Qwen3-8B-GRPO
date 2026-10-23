import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)
    
    # Initialize DP tables
    M = np.full((len_a + 1, len_b + 1), -np.inf)
    X = np.full((len_a + 1, len_b + 1), -np.inf)
    Y = np.full((len_a + 1, len_b + 1), -np.inf)

    if mode == "global":
        M[0, 0] = 0
        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                # Compute match/mismatch score
                if seq_a[i-1] == seq_b[j-1]:
                    score = match
                else:
                    score = mismatch
                
                # Update M[i][j]
                M[i, j] = max(
                    M[i-1, j-1] + score,
                    X[i-1, j] + gap_extend,
                    Y[i-1, j] + gap_extend
                )
                
                # Update X[i][j]
                X[i, j] = max(
                    X[i-1, j] + gap_extend,
                    M[i-1, j] + gap_open,
                    Y[i-1, j] + gap_open + gap_extend
                )
                
                # Update Y[i][j]
                Y[i, j] = max(
                    Y[i, j-1] + gap_extend,
                    M[i, j-1] + gap_open,
                    X[i, j-1] + gap_open + gap_extend
                )
        
        # Backtrack to find alignment
        i, j = len_a, len_b
        aligned_a, aligned_b = [], []
        while i > 0 and j > 0:
            # Check all three possibilities for M[i][j]
            if M[i, j] == M[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif M[i, j] == X[i-1, j] + gap_extend:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif M[i, j] == Y[i-1, j] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                j -= 1
            else:
                break
        
        # Add remaining characters
        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append('-')
            i -= 1
        while j > 0:
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            j -= 1
        
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
        return (M[len_a, len_b], aligned_a, aligned_b)
    else:
        # Local alignment
        M[0, 0] = 0
        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                if seq_a[i-1] == seq_b[j-1]:
                    score = match
                else:
                    score = mismatch
                
                M[i, j] = max(
                    M[i-1, j-1] + score,
                    X[i-1, j] + gap_extend,
                    Y[i-1, j] + gap_extend
                )
                
                X[i, j] = max(
                    X[i-1, j] + gap_extend,
                    M[i-1, j] + gap_open,
                    Y[i-1, j] + gap_open + gap_extend
                )
                
                Y[i, j] = max(
                    Y[i, j-1] + gap_extend,
                    M[i, j-1] + gap_open,
                    X[i, j-1] + gap_open + gap_extend
                )
        
        # Find the maximum score
        max_score = -np.inf
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                current_max = max(M[i, j], X[i, j], Y[i, j])
                if current_max > max_score:
                    max_score = current_max
        if max_score <= 0:
            return (0, "", "")
        
        # Find the cell with the maximum score
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if M[i, j] == max_score or X[i, j] == max_score or Y[i, j] == max_score:
                    i_start, j_start = i, j
                    break
        
        # Backtrack
        i, j = i_start, j_start
        aligned_a, aligned_b = [], []
        while i > 0 and j > 0:
            # Check all three possibilities for M[i][j]
            if M[i, j] == M[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif M[i, j] == X[i-1, j] + gap_extend:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif M[i, j] == Y[i-1, j] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                j -= 1
            else:
                break
            # Stop at the start of the alignment
            if M[i, j] == 0 or X[i, j] == 0 or Y[i, j] == 0:
                break
        
        # Add remaining characters
        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append('-')
            i -= 1
        while j > 0:
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            j -= 1
        
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
        return (max_score, aligned_a, aligned_b)