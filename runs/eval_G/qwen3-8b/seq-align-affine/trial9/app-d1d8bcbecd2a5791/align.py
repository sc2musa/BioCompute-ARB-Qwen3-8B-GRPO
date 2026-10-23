import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    m, n = len(seq_a), len(seq_b)
    INF = float('-inf')
    
    # Initialize DP tables
    dp0 = [[INF] * (n+1) for _ in range(m+1)]
    dp1 = [[INF] * (n+1) for _ in range(m+1)]
    dp2 = [[INF] * (n+1) for _ in range(m+1)]
    
    dp0[0][0] = 0
    
    # Fill DP tables
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 and j == 0:
                continue
            
            # Compute dp0[i][j]
            if i > 0 and j > 0:
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                dp0[i][j] = max(
                    dp0[i-1][j-1] + score,
                    dp1[i-1][j] + score,
                    dp2[i-1][j] + score
                )
            
            # Compute dp1[i][j]
            if i > 0:
                if j > 0:
                    dp1[i][j] = max(
                        dp1[i-1][j] + gap_extend,
                        dp0[i-1][j] + gap_open
                    )
                else:
                    dp1[i][j] = dp0[i][j-1] + gap_open
            else:
                if j > 0:
                    dp1[i][j] = dp0[i][j-1] + gap_open
            
            # Compute dp2[i][j]
            if j > 0:
                if i > 0:
                    dp2[i][j] = max(
                        dp2[i][j-1] + gap_extend,
                        dp0[i][j-1] + gap_open
                    )
                else:
                    dp2[i][j] = dp0[i][j-1] + gap_open
    
    # Determine final score
    if mode == "global":
        score = dp0[m][n]
        aligned_a, aligned_b = backtrack_global(m, n, seq_a, seq_b, dp0, dp1, dp2, match, mismatch, gap_open, gap_extend)
    else:  # local mode
        # Flatten all DP tables to find max_score
        all_values = []
        for dp in [dp0, dp1, dp2]:
            for row in dp:
                all_values.extend(row)
        max_score = max(all_values)
        if max_score <= 0:
            return (0, "", "")
        score = max_score
        aligned_a, aligned_b = backtrack_local(m, n, seq_a, seq_b, dp0, dp1, dp2, match, mismatch, gap_open, gap_extend)
    
    # Ensure score is integer
    return (int(score), aligned_a, aligned_b)


def backtrack_global(m, n, seq_a, seq_b, dp0, dp1, dp2, match, mismatch, gap_open, gap_extend):
    i, j = m, n
    aligned_a = []
    aligned_b = []
    
    while i > 0 and j > 0:
        if dp0[i][j] == dp0[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif dp0[i][j] == dp1[i-1][j] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif dp0[i][j] == dp2[i-1][j] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        else:
            # Handle gaps
            if dp1[i][j] == dp1[i-1][j] + gap_extend:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif dp1[i][j] == dp0[i-1][j] + gap_open:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
            elif dp2[i][j] == dp2[i][j-1] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                j -= 1
            else:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
    
    # Handle remaining characters
    while i > 0:
        aligned_a.append(seq_a[i-1])
        aligned_b.append("-")
        i -= 1
    while j > 0:
        aligned_a.append("-")
        aligned_b.append(seq_b[j-1])
        j -= 1
    
    return ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b))


def backtrack_local(m, n, seq_a, seq_b, dp0, dp1, dp2, match, mismatch, gap_open, gap_extend):
    max_val = 0
    max_i, max_j, max_state = 0, 0, 0
    
    # Find the cell with maximum score
    for i in range(m+1):
        for j in range(n+1):
            if dp0[i][j] > max_val:
                max_val = dp0[i][j]
                max_i, max_j, max_state = i, j, 0
            if dp1[i][j] > max_val:
                max_val = dp1[i][j]
                max_i, max_j, max_state = i, j, 1
            if dp2[i][j] > max_val:
                max_val = dp2[i][j]
                max_i, max_j, max_state = i, j, 2
    
    # Backtrack from the max cell
    i, j = max_i, max_j
    aligned_a = []
    aligned_b = []
    
    while i > 0 or j > 0:
        if max_state == 0:
            if dp0[i][j] == dp0[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif dp0[i][j] == dp1[i-1][j] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif dp0[i][j] == dp2[i-1][j] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            else:
                # Handle gaps
                if dp1[i][j] == dp1[i-1][j] + gap_extend:
                    aligned_a.append("-")
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                elif dp1[i][j] == dp0[i-1][j] + gap_open:
                    aligned_a.append("-")
                    aligned_b.append(seq_b[j-1])
                    j -= 1
                elif dp2[i][j] == dp2[i][j-1] + gap_extend:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    j -= 1
                else:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append("-")
                    i -= 1
        elif max_state == 1:
            # Handle state 1
            if dp1[i][j] == dp1[i-1][j] + gap_extend:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif dp1[i][j] == dp0[i-1][j] + gap_open:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
        elif max_state == 2:
            # Handle state 2
            if dp2[i][j] == dp2[i][j-1] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                j -= 1
            elif dp2[i][j] == dp0[i][j-1] + gap_open:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
        # Determine previous state
        if max_state == 0:
            if dp0[i][j] == dp0[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                max_state = 0
            elif dp0[i][j] == dp1[i-1][j] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                max_state = 1
            elif dp0[i][j] == dp2[i-1][j] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                max_state = 2
            else:
                if dp1[i][j] == dp1[i-1][j] + gap_extend:
                    max_state = 1
                elif dp1[i][j] == dp0[i-1][j] + gap_open:
                    max_state = 0
                elif dp2[i][j] == dp2[i][j-1] + gap_extend:
                    max_state = 2
                else:
                    max_state = 0
        elif max_state == 1:
            if dp1[i][j] == dp1[i-1][j] + gap_extend:
                max_state = 1
            elif dp1[i][j] == dp0[i-1][j] + gap_open:
                max_state = 0
        elif max_state == 2:
            if dp2[i][j] == dp2[i][j-1] + gap_extend:
                max_state = 2
            elif dp2[i][j] == dp0[i][j-1] + gap_open:
                max_state = 0
    
    return ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b))
