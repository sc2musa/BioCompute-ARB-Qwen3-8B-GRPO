"""Reference solution: Gotoh pairwise alignment with affine gap penalties."""

NEG_INF = float("-inf")

# Traceback states
STATE_M, STATE_X, STATE_Y = 0, 1, 2


def _sub(a_char, b_char, match, mismatch):
    return match if a_char == b_char else mismatch


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b).

    A gap of length k scores gap_open + (k - 1) * gap_extend.
    M: column (a_i, b_j); X: column (a_i, '-'); Y: column ('-', b_j).
    """
    if mode not in ("global", "local"):
        raise ValueError(f"unknown mode: {mode}")
    is_local = mode == "local"
    n, m = len(seq_a), len(seq_b)

    M = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    X = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    Y = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    # Pointers: which previous state each cell came from (-1 = alignment start)
    pM = [[-1] * (m + 1) for _ in range(n + 1)]
    pX = [[-1] * (m + 1) for _ in range(n + 1)]
    pY = [[-1] * (m + 1) for _ in range(n + 1)]

    M[0][0] = 0.0
    if not is_local:
        for i in range(1, n + 1):
            X[i][0] = gap_open + (i - 1) * gap_extend
            pX[i][0] = STATE_M if i == 1 else STATE_X
        for j in range(1, m + 1):
            Y[0][j] = gap_open + (j - 1) * gap_extend
            pY[0][j] = STATE_M if j == 1 else STATE_Y

    best_score, best_cell = 0.0, (0, 0)
    for i in range(1, n + 1):
        a_char = seq_a[i - 1]
        Mi, Xi, Yi = M[i], X[i], Y[i]
        Mp, Xp, Yp = M[i - 1], X[i - 1], Y[i - 1]
        for j in range(1, m + 1):
            # M: diagonal move
            cands = (Mp[j - 1], Xp[j - 1], Yp[j - 1])
            k = max(range(3), key=cands.__getitem__)
            prev = cands[k]
            if is_local and prev <= 0.0:
                prev, k = 0.0, -1
            Mi[j] = prev + _sub(a_char, seq_b[j - 1], match, mismatch)
            pM[i][j] = k
            # X: gap in b (consume a_i)
            cands = (Mp[j] + gap_open, Xp[j] + gap_extend, Yp[j] + gap_open)
            k = max(range(3), key=cands.__getitem__)
            Xi[j], pX[i][j] = cands[k], k
            # Y: gap in a (consume b_j)
            cands = (Mi[j - 1] + gap_open, Xi[j - 1] + gap_open, Yi[j - 1] + gap_extend)
            k = max(range(3), key=cands.__getitem__)
            Yi[j], pY[i][j] = cands[k], k
            if is_local and Mi[j] > best_score:
                best_score, best_cell = Mi[j], (i, j)

    if is_local:
        if best_score <= 0.0:
            return 0, "", ""
        i, j = best_cell
        state = STATE_M
        score = best_score
    else:
        if n == 0 and m == 0:
            return 0, "", ""
        finals = (M[n][m], X[n][m], Y[n][m])
        state = max(range(3), key=finals.__getitem__)
        score = finals[state]
        i, j = n, m

    out_a, out_b = [], []
    while state != -1 and (i > 0 or j > 0):
        if state == STATE_M:
            out_a.append(seq_a[i - 1])
            out_b.append(seq_b[j - 1])
            state = pM[i][j]
            i, j = i - 1, j - 1
        elif state == STATE_X:
            out_a.append(seq_a[i - 1])
            out_b.append("-")
            state = pX[i][j]
            i -= 1
        else:
            out_a.append("-")
            out_b.append(seq_b[j - 1])
            state = pY[i][j]
            j -= 1
    return int(score), "".join(reversed(out_a)), "".join(reversed(out_b))
