def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)
    
    # Initialize DP tables
    if mode == "global":
        dp = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        dpA = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        dpB = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        
        dp[0][0] = 0
        for i in range(1, len_a + 1):
            dpB[i][0] = gap_open + (i-1)*gap_extend
        for j in range(1, len_b + 1):
            dpA[0][j] = gap_open + (j-1)*gap_extend
        
        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                if seq_a[i-1] == seq_b[j-1]:
                    score = match
                else:
                    score = mismatch
                
                dp[i][j] = max(dp[i-1][j-1] + score, dpA[i-1][j] + score, dpB[i-1][j] + score)
                dpA[i][j] = max(dpA[i-1][j] + gap_extend, dpB[i-1][j] + gap_open, dp[i-1][j] + gap_open)
                dpB[i][j] = max(dpB[i][j-1] + gap_extend, dp[i][j-1] + gap_open, dpA[i][j-1] + gap_open)
        
        score = dp[len_a][len_b]
        aligned_a, aligned_b = [], []
        i, j = len_a, len_b
        
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                score_current = match
            else:
                score_current = mismatch
            
            if dp[i][j] == dp[i-1][j-1] + score_current:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif dp[i][j] == dpA[i-1][j] + score_current:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
            elif dp[i][j] == dpB[i-1][j] + score_current:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            else:
                break
        
        return (score, ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))
    
    # Local mode implementation
    elif mode == "local":
        dp = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        dpA = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        dpB = [[-float('inf')] * (len_b + 1) for _ in range(len_a + 1)]
        
        dp[0][0] = 0
        for i in range(1, len_a + 1):
            dpB[i][0] = gap_open + (i-1)*gap_extend
        for j in range(1, len_b + 1):
            dpA[0][j] = gap_open + (j-1)*gap_extend
        
        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                if seq_a[i-1] == seq_b[j-1]:
                    score = match
                else:
                    score = mismatch
                
                dp[i][j] = max(dp[i-1][j-1] + score, dpA[i-1][j] + score, dpB[i-1][j] + score, 0)
                dpA[i][j] = max(dpA[i-1][j] + gap_extend, dpB[i-1][j] + gap_open, dp[i-1][j] + gap_open, 0)
                dpB[i][j] = max(dpB[i][j-1] + gap_extend, dp[i][j-1] + gap_open, dpA[i][j-1] + gap_open, 0)
        
        # Find maximum score
        max_score = max(max(row) for row in [dp[i] for i in range(len_a+1)] + [dpA[i] for i in range(len_a+1)] + [dpB[i] for i in range(len_a+1)])
        if max_score < 0:
            return (0, "", "")
        
        # Find the cell with maximum score
        for i in range(len_a+1):
            for j in range(len_b+1):
                if dp[i][j] == max_score:
                    i_max, j_max = i, j
                    break
            else:
                continue
            break
        
        aligned_a, aligned_b = [], []
        i, j = i_max, j_max
        
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                score_current = match
            else:
                score_current = mismatch
            
            if dp[i][j] == dp[i-1][j-1] + score_current:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif dp[i][j] == dpA[i-1][j] + score_current:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                j -= 1
            elif dp[i][j] == dpB[i-1][j] + score_current:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                i -= 1
            else:
                break
        
        return (max_score, ''.join(reversed(aligned_a)), ''.join(reversed(aligned_b)))
    
    else:
        raise ValueError("Invalid mode. Use 'global' or 'local'.")

# Run self-check
import json
with open("$ROOT/runs/eval_round4/rft4_G/seq-align-affine/trial16/app-850b9a8a24bc5e1c/examples.json") as f:
    examples = json.load(f)

for example in examples:
    seq_a, seq_b, mode, match, mismatch, gap_open, gap_extend, expected_score = example
    score, aligned_a, aligned_b = align(seq_a, seq_b, mode, match, mismatch, gap_open, gap_extend)
    assert score == expected_score, f"Failed for {example}"

print("All examples passed!")