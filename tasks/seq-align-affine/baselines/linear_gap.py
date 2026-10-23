"""Plausible-but-wrong baseline: Needleman-Wunsch/Smith-Waterman with *linear* gaps.

Every gap column scores gap_open, i.e. the affine extension is ignored. Used to check that
the verifier rejects a common mistake (expected: fails most cases).
"""


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    n, m = len(seq_a), len(seq_b)
    local = mode == "local"
    H = [[0] * (m + 1) for _ in range(n + 1)]
    if not local:
        for i in range(1, n + 1):
            H[i][0] = i * gap_open
        for j in range(1, m + 1):
            H[0][j] = j * gap_open
    best, cell = 0, (n, m)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if seq_a[i - 1] == seq_b[j - 1] else mismatch
            v = max(H[i - 1][j - 1] + s, H[i - 1][j] + gap_open, H[i][j - 1] + gap_open)
            H[i][j] = max(v, 0) if local else v
            if local and H[i][j] > best:
                best, cell = H[i][j], (i, j)
    i, j = cell if local else (n, m)
    out_a, out_b = [], []
    while i > 0 or j > 0:
        if local and H[i][j] == 0:
            break
        s = match if i and j and seq_a[i - 1] == seq_b[j - 1] else mismatch
        if i and j and H[i][j] == H[i - 1][j - 1] + s:
            out_a.append(seq_a[i - 1]); out_b.append(seq_b[j - 1]); i -= 1; j -= 1
        elif i and H[i][j] == H[i - 1][j] + gap_open:
            out_a.append(seq_a[i - 1]); out_b.append("-"); i -= 1
        else:
            out_a.append("-"); out_b.append(seq_b[j - 1]); j -= 1
    return (best if local else H[n][m]), "".join(reversed(out_a)), "".join(reversed(out_b))
