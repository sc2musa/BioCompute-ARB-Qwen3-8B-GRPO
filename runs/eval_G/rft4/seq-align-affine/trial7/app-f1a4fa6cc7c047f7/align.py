def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    m, n = len(seq_a), len(seq_b)
    
    # Initialize DP tables
    S = [[0]*(n+1) for _ in range(m+1)]
    X = [[-float('inf')]*(n+1) for _ in range(m+1)]
    Y = [[-float('inf')]*(n+1) for _ in range(m+1)]
    
    # Initialize for i=0, j=0
    S[0][0] = 0
    
    # Initialize for i=0, j>0
    for j in range(1, n+1):
        Y[0][j] = gap_open + (j-1)*gap_extend
    
    # Initialize for j=0, i>0
    for i in range(1, m+1):
        X[i][0] = gap_open + (i-1)*gap_extend
    
    # Fill DP tables
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Compute S[i][j]
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            if mode == "local":
                S[i][j] = max(S[i-1][j-1] + score, X[i-1][j] + score, Y[i][j-1] + score, 0)
            else:
                S[i][j] = max(S[i-1][j-1] + score, X[i-1][j] + score, Y[i][j-1] + score)
            
            # Compute X[i][j]
            X[i][j] = max(X[i-1][j] + gap_extend, Y[i-1][j] + gap_open)
            
            # Compute Y[i][j]
            Y[i][j] = max(Y[i][j-1] + gap_extend, X[i][j-1] + gap_open)
    
    if mode == "global":
        score = S[m][n]
        aligned_a, aligned_b = [], []
        i, j = m, n
        while i > 0 and j > 0:
            if S[i][j] == S[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif S[i][j] == X[i-1][j] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            elif S[i][j] == Y[i][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
            else:
                break
        
        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append('-')
            i -= 1
        while j > 0:
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            j -= 1
        
        return (score, ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))
    else:
        # Local mode
        max_score = -float('inf')
        max_i, max_j = 0, 0
        for i in range(m+1):
            for j in range(n+1):
                current = max(S[i][j], X[i][j], Y[i][j])
                if current > max_score:
                    max_score = current
                    max_i, max_j = i, j
        
        if max_score == 0:
            return (0, "", "")
        
        aligned_a, aligned_b = [], []
        i, j = max_i, max_j
        while i > 0 and j > 0:
            # Check which table the current value came from
            if S[i][j] == max_score:
                # Check if this cell was from S[i][j]
                if S[i][j] - (match if seq_a[i-1] == seq_b[j-1] else mismatch) == S[i-1][j-1]:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                elif S[i][j] - (match if seq_a[i-1] == seq_b[j-1] else mismatch) == X[i-1][j]:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    i -= 1
                elif S[i][j] - (match if seq_a[i-1] == seq_b[j-1] else mismatch) == Y[i][j-1]:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    j -= 1
                elif S[i][j] - (match if seq_a[i-1] == seq_b[j-1] else mismatch) == 0:
                    # Start of new alignment
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                else:
                    break
            elif X[i][j] == max_score:
                # Check if this cell was from X[i][j]
                if X[i][j] == X[i-1][j] + gap_extend:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    i -= 1
                elif X[i][j] == Y[i-1][j] + gap_open:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    i -= 1
                else:
                    break
            elif Y[i][j] == max_score:
                # Check if this cell was from Y[i][j]
                if Y[i][j] == Y[i][j-1] + gap_extend:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    j -= 1
                elif Y[i][j] == X[i][j-1] + gap_open:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    j -= 1
                else:
                    break
            else:
                break
        
        return (max_score, ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))
