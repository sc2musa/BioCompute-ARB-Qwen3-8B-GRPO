import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)
    
    # Handle empty sequences
    if len_a == 0 and len_b == 0:
        return (0, "", "")
    elif len_a == 0:
        return (gap_open + (len_b - 1) * gap_extend, 
                "-" * len_b, seq_b)
    elif len_b == 0:
        return (gap_open + (len_a - 1) * gap_extend, 
                seq_a, "-" * len_a)
    
    # Initialize DP and path tables
    if mode == "global":
        dp = [[[0]*3 for _ in range(len_b+1)] for _ in range(len_a+1)]
        path = [[0]*(len_b+1) for _ in range(len_a+1)]
        
        # Initialize first row (i=0)
        for j in range(1, len_b+1):
            dp[0][j][1] = gap_open + (j-1)*gap_extend
            dp[0][j][0] = dp[0][j][2] = float('-inf')
        
        # Initialize first column (j=0)
        for i in range(1, len_a+1):
            dp[i][0][2] = gap_open + (i-1)*gap_extend
            dp[i][0][0] = dp[i][0][1] = float('-inf')
        
        # Fill DP table
        for i in range(1, len_a+1):
            for j in range(1, len_b+1):
                # State 0: match/mismatch
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                option0 = max(
                    dp[i-1][j-1][0] + score,
                    dp[i-1][j][1] + score,
                    dp[i][j-1][2] + score
                )
                dp[i][j][0] = option0
                
                # State 1: gap in A
                option1 = max(
                    dp[i-1][j][0] + gap_open,
                    dp[i-1][j][1] + gap_extend
                )
                dp[i][j][1] = option1
                
                # State 2: gap in B
                option2 = max(
                    dp[i][j-1][0] + gap_open,
                    dp[i][j-1][2] + gap_extend
                )
                dp[i][j][2] = option2
                
                # Determine path
                max_val = max(dp[i][j])
                if dp[i][j][0] == max_val:
                    path[i][j] = 0
                elif dp[i][j][1] == max_val:
                    path[i][j] = 1
                else:
                    path[i][j] = 2
        
        # Get final score
        max_score = max(dp[len_a][len_b])
        
        # Backtrack
        aligned_a, aligned_b = [], []
        i, j = len_a, len_b
        while i > 0 or j > 0:
            state = path[i][j]
            if state == 0:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif state == 1:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
            else:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                j -= 1
        
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
        return (max_score, aligned_a, aligned_b)
    
    # Local mode implementation
    else:
        dp = [[[0]*3 for _ in range(len_b+1)] for _ in range(len_a+1)]
        path = [[0]*(len_b+1) for _ in range(len_a+1)]
        
        # Initialize first row (i=0) and first column (j=0) with zeros
        for j in range(len_b+1):
            dp[0][j][0] = 0
            dp[0][j][1] = float('-inf')
            dp[0][j][2] = float('-inf')
        for i in range(len_a+1):
            dp[i][0][0] = 0
            dp[i][0][1] = float('-inf')
            dp[i][0][2] = float('-inf')
        
        # Fill DP table
        for i in range(1, len_a+1):
            for j in range(1, len_b+1):
                # State 0: match/mismatch
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                option0 = max(
                    dp[i-1][j-1][0] + score,
                    dp[i-1][j][1] + score,
                    dp[i][j-1][2] + score
                )
                dp[i][j][0] = option0
                
                # State 1: gap in A
                option1 = max(
                    dp[i-1][j][0] + gap_open,
                    dp[i-1][j][1] + gap_extend
                )
                dp[i][j][1] = option1
                
                # State 2: gap in B
                option2 = max(
                    dp[i][j-1][0] + gap_open,
                    dp[i][j-1][2] + gap_extend
                )
                dp[i][j][2] = option2
                
                # Determine path
                max_val = max(dp[i][j])
                if dp[i][j][0] == max_val:
                    path[i][j] = 0
                elif dp[i][j][1] == max_val:
                    path[i][j] = 1
                else:
                    path[i][j] = 2
        
        # Find maximum score
        max_score = 0
        max_i, max_j = 0, 0
        for i in range(len_a+1):
            for j in range(len_b+1):
                if dp[i][j][0] > max_score:
                    max_score = dp[i][j][0]
                    max_i, max_j = i, j
                elif dp[i][j][1] > max_score:
                    max_score = dp[i][j][1]
                    max_i, max_j = i, j
                elif dp[i][j][2] > max_score:
                    max_score = dp[i][j][2]
                    max_i, max_j = i, j
        
        if max_score <= 0:
            return (0, "", "")
        
        # Backtrack
        aligned_a, aligned_b = [], []
        i, j = max_i, max_j
        while i > 0 or j > 0:
            state = path[i][j]
            if state == 0:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif state == 1:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
            else:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                j -= 1
        
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
        return (max_score, aligned_a, aligned_b)