def align(seq_a: str, seq_b: str, mode: str = "global",
              match: int = 2, mismatch: int = -3,
              gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    # Handle empty sequences
    if not seq_a and not seq_b:
        return (0, "", "")
    len_a = len(seq_a)
    len_b = len(seq_b)
    # Initialize the DP matrix
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    # Initialize the path matrix
    path = [[None] * (len_b + 1) for _ in range(len_a + 1)]
    # Set the first row (i=0, j >=1)
    for j in range(1, len_b + 1):
        path[0][j] = 'B'  # Gap in seq_b
    # Set the first column (j=0, i >=1)
    for i in range(1, len_a + 1):
        path[i][0] = 'A'  # Gap in seq_a
    # Initialize the first row and column
    for j in range(1, len_b + 1):
        dp[0][j] = gap_open + (j - 1) * gap_extend
    for i in range(1, len_a + 1):
        dp[i][0] = gap_open + (i - 1) * gap_extend
    # Fill the DP matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate the match/mismatch score
            if seq_a[i-1] == seq_b[j-1]:
                match_score = match
            else:
                match_score = mismatch
            # Calculate the three possible scores
            score1 = dp[i-1][j-1] + match_score
            score2 = dp[i-1][j] + gap_extend
            score3 = dp[i][j-1] + gap_extend
            # Find the maximum score and the corresponding direction
            max_score = max(score1, score2, score3)
            if max_score == score1:
                path[i][j] = 'M'
            elif max_score == score2:
                path[i][j] = 'A'
            else:
                path[i][j] = 'B'
            dp[i][j] = max_score
    # Determine the maximum score and the corresponding position
    if mode == "global":
        max_score = dp[len_a][len_b]
        i, j = len_a, len_b
    else:  # mode == "local"
        max_score = -float('inf')
        max_i, max_j = 0, 0
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if dp[i][j] > max_score:
                    max_score = dp[i][j]
                    max_i, max_j = i, j
        # If the maximum score is negative, return (0, "", "")
        if max_score < 0:
            return (0, "", "")
        i, j = max_i, max_j
    # Backtrack to reconstruct the alignment
    aligned_a = []
    aligned_b = []
    while i > 0 or j > 0:
        direction = path[i][j]
        if direction == 'M':
            # Match, so add the characters
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif direction == 'A':
            # Gap in seq_a, so add '-' to aligned_a
            if j > 0:
                aligned_b.append(seq_b[j-1])
            else:
                aligned_b.append('')
            aligned_a.append('-')
            i -= 1
        else:  # direction == 'B'
            # Gap in seq_b, so add '-' to aligned_b
            if i > 0:
                aligned_a.append(seq_a[i-1])
            else:
                aligned_a.append('')
            aligned_b.append('-')
            j -= 1
    # Reverse the aligned sequences
    aligned_a = ''.join(reversed(aligned_a))
    aligned_b = ''.join(reversed(aligned_b))
    # Return the score, aligned_a, and aligned_b
    return (max_score, aligned_a, aligned_b)