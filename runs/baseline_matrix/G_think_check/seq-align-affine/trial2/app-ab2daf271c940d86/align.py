import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    if not seq_a and not seq_b:
        return (0, "", "")
    if not seq_a:
        if mode == "global":
            score = gap_open + (len(seq_b) - 1)*gap_extend
            return (score, "-"*len(seq_b), seq_b)
        else:
            score = gap_open + (len(seq_b) - 1)*gap_extend
            return (score, "-"*len(seq_b), seq_b) if score > 0 else (0, "", "")
    if not seq_b:
        if mode == "global":
            score = gap_open + (len(seq_a) - 1)*gap_extend
            return (score, seq_a, "-"*len(seq_a))
        else:
            score = gap_open + (len(seq_a) - 1)*gap_extend
            return (score, seq_a, "-"*len(seq_a)) if score > 0 else (0, "", "")
    len_a, len_b = len(seq_a), len(seq_b)
    INF = float('-inf')
    state0 = [[INF]*(len_b+1) for _ in range(len_a+1)]
    state1 = [[INF]*(len_b+1) for _ in range(len_a+1)]
    state2 = [[INF]*(len_b+1) for _ in range(len_a+1)]
    state0[0][0] = 0
    for j in range(1, len_b+1):
        state1[0][j] = gap_open + (j-1)*gap_extend
    for i in range(1, len_a+1):
        state2[i][0] = gap_open + (i-1)*gap_extend
    for i in range(1, len_a+1):
        for j in range(1, len_b+1):
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            option1 = state0[i-1][j-1] + score
            option2 = state1[i-1][j] + score
            option3 = state2[i][j-1] + score
            state0[i][j] = max(option1, option2, option3)
            option1 = state1[i-1][j] + gap_extend
            option2 = state0[i-1][j] + gap_open
            state1[i][j] = max(option1, option2)
            option1 = state2[i][j-1] + gap_extend
            option2 = state0[i][j-1] + gap_open
            state2[i][j] = max(option1, option2)
    if mode == "global":
        score = state0[len_a][len_b]
        aligned_a, aligned_b = backtrack(state0, state1, state2, seq_a, seq_b, mode, len_a, len_b, 0)
        return (score, aligned_a, aligned_b)
    else:
        max_score = -float('inf')
        max_i, max_j, max_state = 0, 0, 0
        for i in range(len_a+1):
            for j in range(len_b+1):
                if state0[i][j] > max_score:
                    max_score = state0[i][j]
                    max_i, max_j, max_state = i, j, 0
                if state1[i][j] > max_score:
                    max_score = state1[i][j]
                    max_i, max_j, max_state = i, j, 1
                if state2[i][j] > max_score:
                    max_score = state2[i][j]
                    max_i, max_j, max_state = i, j, 2
        if max_score <= 0:
            return (0, "", "")
        aligned_a, aligned_b = backtrack(state0, state1, state2, seq_a, seq_b, mode, max_i, max_j, max_state)
        return (max_score, aligned_a, aligned_b)

def backtrack(state0, state1, state2, seq_a, seq_b, mode, i, j, state):
    aligned_a = []
    aligned_b = []
    while i > 0 and j > 0:
        if state == 0:
            if seq_a[i-1] == seq_b[j-1]:
                score = match
            else:
                score = mismatch
            option1 = state0[i-1][j-1] + score
            option2 = state1[i-1][j] + score
            option3 = state2[i][j-1] + score
            if state0[i][j] == option1:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
                state = 0
            elif state0[i][j] == option2:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
                state = 1
            elif state0[i][j] == option3:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                j -= 1
                state = 2
        elif state == 1:
            option1 = state1[i-1][j] + gap_extend
            option2 = state0[i-1][j] + gap_open
            if state1[i][j] == option1:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
                state = 1
            elif state1[i][j] == option2:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
                state = 0
        elif state == 2:
            option1 = state2[i][j-1] + gap_extend
            option2 = state0[i][j-1] + gap_open
            if state2[i][j] == option1:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                j -= 1
                state = 2
            elif state2[i][j] == option2:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                j -= 1
                state = 0
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
    return (''.join(aligned_a), ''.join(aligned_b))