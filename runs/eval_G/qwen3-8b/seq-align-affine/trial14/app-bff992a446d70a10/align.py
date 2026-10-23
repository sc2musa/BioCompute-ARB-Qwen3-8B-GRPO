import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)
    INF = float('-inf')
    
    if mode == "global":
        # Initialize DP tables
        score = [[INF]*(len_b+1) for _ in range(len_a+1)]
        score_a = [[INF]*(len_b+1) for _ in range(len_a+1)]
        score_b = [[INF]*(len_b+1) for _ in range(len_a+1)]
        score[0][0] = 0
        
        # Fill first row and column
        for j in range(1, len_b+1):
            score_b[0][j] = max(score_b[0][j-1] + gap_extend, score[0][j-1] + gap_open)
        for i in range(1, len_a+1):
            score_a[i][0] = max(score_a[i-1][0] + gap_extend, score[i-1][0] + gap_open)
        
        # Fill DP tables
        for i in range(1, len_a+1):
            for j in range(1, len_b+1):
                match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                score[i][j] = max(
                    score[i-1][j-1] + match_score,
                    score_a[i-1][j] + gap_extend,
                    score_b[i-1][j] + gap_extend
                )
                score_a[i][j] = max(
                    score[i-1][j] + gap_open,
                    score_a[i-1][j] + gap_extend
                )
                score_b[i][j] = max(
                    score[i][j-1] + gap_open,
                    score_b[i][j-1] + gap_extend
                )
        
        # Backtrack
        aligned_a, aligned_b = [], []
        i, j = len_a, len_b
        while i > 0 and j > 0:
            if score[i][j] == score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif score[i][j] == score_a[i-1][j] + gap_extend:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif score[i][j] == score_b[i-1][j] + gap_extend:
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
        
        aligned_a.reverse()
        aligned_b.reverse()
        return (score[len_a][len_b], ''.join(aligned_a), ''.join(aligned_b))
    
    elif mode == "local":
        # Initialize DP tables with 0
        score = [[0]*(len_b+1) for _ in range(len_a+1)]
        score_a = [[0]*(len_b+1) for _ in range(len_a+1)]
        score_b = [[0]*(len_b+1) for _ in range(len_a+1)]
        
        # Fill DP tables
        for i in range(len_a+1):
            for j in range(len_b+1):
                if i == 0 or j == 0:
                    # First row/column
                    if i == 0:
                        score_b[i][j] = max(score_b[i][j-1] + gap_extend, score[i][j-1] + gap_open)
                    if j == 0:
                        score_a[i][j] = max(score_a[i-1][j] + gap_extend, score[i-1][j] + gap_open)
                else:
                    match_score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                    score[i][j] = max(
                        score[i-1][j-1] + match_score,
                        score_a[i-1][j] + gap_extend,
                        score_b[i-1][j] + gap_extend,
                        0
                    )
                    score_a[i][j] = max(
                        score[i-1][j] + gap_open,
                        score_a[i-1][j] + gap_extend,
                        0
                    )
                    score_b[i][j] = max(
                        score[i][j-1] + gap_open,
                        score_b[i][j-1] + gap_extend,
                        0
                    )
        
        # Find maximum score
        max_score = 0
        max_i, max_j = 0, 0
        for i in range(len_a+1):
            for j in range(len_b+1):
                if score[i][j] > max_score:
                    max_score = score[i][j]
                    max_i, max_j = i, j
        
        # Backtrack
        aligned_a, aligned_b = [], []
        i, j = max_i, max_j
        while i > 0 and j > 0:
            if score[i][j] == score[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif score[i][j] == score_a[i-1][j] + gap_extend:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif score[i][j] == score_b[i-1][j] + gap_extend:
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
        
        aligned_a.reverse()
        aligned_b.reverse()
        return (max_score, ''.join(aligned_a), ''.join(aligned_b))
    
    else:
        raise ValueError("Unsupported mode")

# Run self-check
import json
with open("$ROOT/runs/eval_round4/base_G/seq-align-affine/trial14/app-bff992a446d70a10/examples.json") as f:
    examples = json.load(f)

for ex in examples:
    seq_a = ex["seq_a"]
    seq_b = ex["seq_b"]
    mode = ex["mode"]
    expected_score = ex["expected_score"]
    match = ex.get("match", 2)
    mismatch = ex.get("mismatch", -3)
    gap_open = ex.get("gap_open", -5)
    gap_extend = ex.get("gap_extend", -2)
    score, _, _ = align(seq_a, seq_b, mode, match, mismatch, gap_open, gap_extend)
    assert score == expected_score, f"Failed for {ex}"

print("All examples passed!")