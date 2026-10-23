import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""

    if not seq_a and not seq_b:
        return (0, "", "")
    elif not seq_a:
        if mode == "global":
            return (gap_open, "-" * len(seq_b), seq_b)
        else:
            return (0, "", "")
    elif not seq_b:
        if mode == "global":
            return (gap_open, seq_a, "-" * len(seq_a))
        else:
            return (0, "", "")

    len_a, len_b = len(seq_a), len(seq_b)

    if mode == "global":
        M = np.full((len_a + 1, len_b + 1), -np.inf)
        D = np.full((len_a + 1, len_b + 1), -np.inf)
        I = np.full((len_a + 1, len_b + 1), -np.inf)
        M[0, 0] = 0
    else:
        M = np.full((len_a + 1, len_b + 1), -np.inf)
        D = np.full((len_a + 1, len_b + 1), -np.inf)
        I = np.full((len_a + 1, len_b + 1), -np.inf)
        S = np.full((len_a + 1, len_b + 1), -np.inf)
        S[0, 0] = 0

    for i in range(len_a + 1):
        for j in range(len_b + 1):
            if i == 0 and j == 0:
                continue
            if mode == "global":
                if i > 0 and j > 0:
                    score_m = match if seq_a[i-1] == seq_b[j-1] else mismatch
                    M[i, j] = max(M[i-1, j-1] + score_m, D[i-1, j] + gap_extend, I[i, j-1] + gap_extend)
                if i > 0:
                    D[i, j] = max(M[i-1, j] + gap_open, D[i-1, j] + gap_extend)
                if j > 0:
                    I[i, j] = max(M[i, j-1] + gap_open, I[i-1, j] + gap_extend)
            else:
                if i > 0 and j > 0:
                    score_m = match if seq_a[i-1] == seq_b[j-1] else mismatch
                    M[i, j] = max(M[i-1, j-1] + score_m, D[i-1, j] + gap_extend, I[i, j-1] + gap_extend)
                if i > 0:
                    D[i, j] = max(M[i-1, j] + gap_open, D[i-1, j] + gap_extend, S[i-1, j] + gap_open)
                if j > 0:
                    I[i, j] = max(M[i, j-1] + gap_open, I[i-1, j] + gap_extend, S[i, j-1] + gap_open)
                if i > 0 or j > 0:
                    S[i, j] = max(M[i, j], S[i-1, j], S[i, j-1])

    if mode == "global":
        score = M[len_a, len_b]
        i, j = len_a, len_b
        aligned_a, aligned_b = [], []
        while i > 0 and j > 0:
            if M[i, j] == M[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif M[i, j] == D[i-1, j] + gap_extend:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif M[i, j] == I[i, j-1] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                j -= 1
            else:
                break
        while i > 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        while j > 0:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
        aligned_a = aligned_a[::-1]
        aligned_b = aligned_b[::-1]
    else:
        max_score = -np.inf
        max_i, max_j = 0, 0
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                current = np.max([M[i, j], D[i, j], I[i, j], S[i, j]])
                if current > max_score:
                    max_score = current
                    max_i, max_j = i, j
        if max_score <= 0:
            return (0, "", "")
        score = max_score
        i, j = max_i, max_j
        aligned_a, aligned_b = [], []
        while i > 0 or j > 0:
            if i > 0 and j > 0 and M[i, j] == M[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif i > 0 and D[i, j] == D[i-1, j] + gap_extend:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif j > 0 and I[i, j] == I[i, j-1] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                j -= 1
            elif i > 0 and S[i, j] == S[i-1, j] + 0:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif j > 0 and S[i, j] == S[i, j-1] + 0:
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            else:
                break
        aligned_a = aligned_a[::-1]
        aligned_b = aligned_b[::-1]
    return (int(score), "".join(aligned_a), "".join(aligned_b))

