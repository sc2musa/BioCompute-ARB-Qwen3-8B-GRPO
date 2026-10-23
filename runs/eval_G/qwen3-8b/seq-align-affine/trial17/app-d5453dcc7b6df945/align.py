import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)
    
    if len_a == 0 and len_b == 0:
        return (0, "", "")
    
    if len_a == 0:
        return (gap_open + (len_b - 1) * gap_extend, "-" * len_b, seq_b)
    if len_b == 0:
        return (gap_open + (len_a - 1) * gap_extend, seq_a, "-" * len_a)
    
    if mode == "global":
        normal = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        gap_a = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        gap_b = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        
        normal[0][0] = 0
        for j in range(1, len_b + 1):
            gap_b[0][j] = gap_open + (j - 1) * gap_extend
        for i in range(1, len_a + 1):
            gap_a[i][0] = gap_open + (i - 1) * gap_extend
    else:
        normal = [[0] * (len_b + 1) for _ in range(len_a + 1)]
        gap_a = [[0] * (len_b + 1) for _ in range(len_a + 1)]
        gap_b = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            
            normal[i][j] = max(
                normal[i-1][j-1] + score,
                gap_a[i-1][j] + gap_extend,
                gap_b[i-1][j] + gap_extend
            )
            
            gap_a[i][j] = max(
                normal[i-1][j] + gap_open,
                gap_a[i-1][j] + gap_extend
            )
            
            gap_b[i][j] = max(
                normal[i][j-1] + gap_open,
                gap_b[i][j-1] + gap_extend
            )
    
    if mode == "global":
        final_score = max(normal[len_a][len_b], gap_a[len_a][len_b], gap_b[len_a][len_b])
        i, j = len_a, len_b
        aligned_a, aligned_b = [], []
        while i > 0 or j > 0:
            if normal[i][j] == final_score:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
                final_score = normal[i][j] if i > 0 and j > 0 else -float('inf')
            elif gap_a[i][j] == final_score:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
                final_score = gap_a[i][j] if i > 0 else -float('inf')
            elif gap_b[i][j] == final_score:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                j -= 1
                final_score = gap_b[i][j] if j > 0 else -float('inf')
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
    else:
        max_score = 0
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                current_max = max(normal[i][j], gap_a[i][j], gap_b[i][j])
                if current_max > max_score:
                    max_score = current_max
        if max_score == 0:
            return (0, "", "")
        
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if max(normal[i][j], gap_a[i][j], gap_b[i][j]) == max_score:
                    break
            else:
                continue
            break
        
        final_score = max_score
        aligned_a, aligned_b = [], []
        while i > 0 or j > 0:
            if normal[i][j] == final_score:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
                final_score = normal[i][j] if i > 0 and j > 0 else -float('inf')
            elif gap_a[i][j] == final_score:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
                final_score = gap_a[i][j] if i > 0 else -float('inf')
            elif gap_b[i][j] == final_score:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                j -= 1
                final_score = gap_b[i][j] if j > 0 else -float('inf')
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
    
    return (final_score, aligned_a, aligned_b)