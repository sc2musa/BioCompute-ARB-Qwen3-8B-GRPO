import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    if not seq_a and not seq_b:
        return (0, "", "")
    
    m = len(seq_a)
    n = len(seq_b)
    
    if mode == "global":
        dp = np.full((m+1, n+1), -np.inf)
        gap_a = np.full((m+1, n+1), -np.inf)
        gap_b = np.full((m+1, n+1), -np.inf)
        dp[0][0] = 0
        
        for j in range(1, n+1):
            gap_b[0][j] = gap_open + (j-1)*gap_extend
        for i in range(1, m+1):
            gap_a[i][0] = gap_open + (i-1)*gap_extend
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                dp[i][j] = max(
                    dp[i-1][j-1] + score,
                    gap_a[i-1][j] + gap_extend,
                    gap_b[i][j-1] + gap_extend
                )
                gap_a[i][j] = max(
                    dp[i-1][j] + gap_open,
                    gap_a[i-1][j] + gap_extend
                )
                gap_b[i][j] = max(
                    dp[i][j-1] + gap_open,
                    gap_b[i][j-1] + gap_extend
                )
        
        score = dp[m][n]
        
        # Backtrack
        aligned_a = []
        aligned_b = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                if dp[i][j] == dp[i-1][j-1] + score:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                elif dp[i][j] == gap_a[i-1][j] + gap_extend:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                elif dp[i][j] == gap_b[i][j-1] + gap_extend:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    j -= 1
            elif i > 0:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            else:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
        
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
        
        return (score, aligned_a, aligned_b)
    
    elif mode == "local":
        dp = np.full((m+1, n+1), 0)
        gap_a = np.full((m+1, n+1), 0)
        gap_b = np.full((m+1, n+1), 0)
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                dp[i][j] = max(
                    dp[i-1][j-1] + score,
                    gap_a[i-1][j] + gap_extend,
                    gap_b[i][j-1] + gap_extend,
                    0
                )
                gap_a[i][j] = max(
                    dp[i-1][j] + gap_open,
                    gap_a[i-1][j] + gap_extend,
                    0
                )
                gap_b[i][j] = max(
                    dp[i][j-1] + gap_open,
                    gap_b[i][j-1] + gap_extend,
                    0
                )
        
        # Find maximum score
        max_score = 0
        max_i, max_j = 0, 0
        for i in range(m+1):
            for j in range(n+1):
                if dp[i][j] > max_score:
                    max_score = dp[i][j]
                    max_i, max_j = i, j
                if gap_a[i][j] > max_score:
                    max_score = gap_a[i][j]
                    max_i, max_j = i, j
                if gap_b[i][j] > max_score:
                    max_score = gap_b[i][j]
                    max_i, max_j = i, j
        
        if max_score <= 0:
            return (0, "", "")
        
        # Backtrack
        aligned_a = []
        aligned_b = []
        i, j = max_i, max_j
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                if dp[i][j] == dp[i-1][j-1] + score:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                elif dp[i][j] == gap_a[i-1][j] + gap_extend:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                elif dp[i][j] == gap_b[i][j-1] + gap_extend:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    j -= 1
            elif i > 0:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            else:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
        
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
        
        return (max_score, aligned_a, aligned_b)
    
    else:
        raise ValueError("Invalid mode.")

# Example usage:
# print(align("ACGT", "ACG", "global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2))
# print(align("ACGT", "ACG", "local", match=2, mismatch=-3, gap_open=-5, gap_extend=-2))
