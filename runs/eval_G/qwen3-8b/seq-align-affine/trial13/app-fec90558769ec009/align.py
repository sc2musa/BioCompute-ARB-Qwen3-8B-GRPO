def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)

    if len_a == 0 and len_b == 0:
        return (0, "", "")

    # Initialize DP table with three states: 0=match/mismatch, 1=gap in A, 2=gap in B
    dp = [[[0]*3 for _ in range(len_b+1)] for _ in range(len_a+1)]

    # Initialize first row and column
    for i in range(len_a+1):
        for j in range(len_b+1):
            if i == 0 and j == 0:
                dp[i][j][0] = 0
                dp[i][j][1] = -float('inf')
                dp[i][j][2] = -float('inf')
            elif i == 0:
                # All gaps in A
                if j > 0:
                    dp[i][j][1] = gap_open + (j-1)*gap_extend
                dp[i][j][0] = -float('inf')
                dp[i][j][2] = -float('inf')
            elif j == 0:
                # All gaps in B
                if i > 0:
                    dp[i][j][2] = gap_open + (i-1)*gap_extend
                dp[i][j][0] = -float('inf')
                dp[i][j][1] = -float('inf')
            else:
                # Compute state 0 (match/mismatch)
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                dp[i][j][0] = dp[i-1][j-1][0] + score

                # Compute state 1 (gap in A)
                option1 = dp[i-1][j][1] + gap_extend
                option2 = max(dp[i-1][j][0], dp[i-1][j][2]) + gap_open
                dp[i][j][1] = max(option1, option2)

                # Compute state 2 (gap in B)
                option1 = dp[i][j-1][2] + gap_extend
                option2 = max(dp[i][j-1][0], dp[i][j-1][1]) + gap_open
                dp[i][j][2] = max(option1, option2)

    # Determine the score based on mode
    if mode == "global":
        # Global alignment must end in state 0
        score = dp[len_a][len_b][0]
        if score <= 0:
            return (0, "", "")
        i, j = len_a, len_b
    else:
        # Find the maximum score in the DP table
        max_score = -float('inf')
        max_i, max_j = 0, 0
        for i in range(len_a+1):
            for j in range(len_b+1):
                for state in range(3):
                    if dp[i][j][state] > max_score:
                        max_score = dp[i][j][state]
                        max_i, max_j = i, j
        if max_score <= 0:
            return (0, "", "")
        score = max_score
        i, j = max_i, max_j

    # Backtrack to find the alignment
    aligned_a = []
    aligned_b = []

    while i > 0 and j > 0:
        # Determine which state we are in
        current_max = -float('inf')
        current_state = 0
        if dp[i][j][0] > current_max:
            current_max = dp[i][j][0]
            current_state = 0
        if dp[i][j][1] > current_max:
            current_max = dp[i][j][1]
            current_state = 1
        if dp[i][j][2] > current_max:
            current_max = dp[i][j][2]
            current_state = 2

        if current_state == 0:
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif current_state == 1:
            aligned_a.append('-')
            aligned_b.append(seq_b[j-1])
            i -= 1
        elif current_state == 2:
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

    # Reverse the aligned sequences
    aligned_a.reverse()
    aligned_b.reverse()

    return (score, ''.join(aligned_a), ''.join(aligned_b))