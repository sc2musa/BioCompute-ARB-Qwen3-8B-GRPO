def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a = len(seq_a)
    len_b = len(seq_b)

    if mode == 'global':
        # Initialize DP tables
        dp = [[0]*(len_b+1) for _ in range(len_a+1)]
        gapA = [[0]*(len_b+1) for _ in range(len_a+1)]
        gapB = [[0]*(len_b+1) for _ in range(len_a+1)]

        # Initialize first row and column
        for j in range(1, len_b+1):
            gapB[0][j] = gap_open + (j-1)*gap_extend
        for i in range(1, len_a+1):
            gapA[i][0] = gap_open + (i-1)*gap_extend

        # Fill the DP tables
        for i in range(1, len_a+1):
            for j in range(1, len_b+1):
                # Compute score for match/mismatch
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                option1 = dp[i-1][j-1] + score
                option2 = gapA[i-1][j] + gap_extend
                option3 = gapB[i][j-1] + gap_extend
                dp[i][j] = max(option1, option2, option3)

                # Compute gapA[i][j]
                option1_gapA = dp[i-1][j] + gap_open
                option2_gapA = gapA[i-1][j] + gap_extend
                gapA[i][j] = max(option1_gapA, option2_gapA)

                # Compute gapB[i][j]
                option1_gapB = dp[i][j-1] + gap_open
                option2_gapB = gapB[i][j-1] + gap_extend
                gapB[i][j] = max(option1_gapB, option2_gapB)

        # Backtrack to find the alignment
        current_i, current_j = len_a, len_b
        aligned_a = []
        aligned_b = []

        while current_i > 0 and current_j > 0:
            # Check if current cell was a match/mismatch
            if dp[current_i][current_j] == dp[current_i-1][current_j-1] + (match if seq_a[current_i-1] == seq_b[current_j-1] else mismatch):
                aligned_a.append(seq_a[current_i-1])
                aligned_b.append(seq_b[current_j-1])
                current_i -= 1
                current_j -= 1
            elif gapA[current_i][current_j] == gapA[current_i-1][current_j] + gap_extend:
                aligned_a.append('-')
                aligned_b.append(seq_b[current_j-1])
                current_i -= 1
            elif gapB[current_i][current_j] == gapB[current_i][current_j-1] + gap_extend:
                aligned_a.append(seq_a[current_i-1])
                aligned_b.append('-')
                current_j -= 1
            else:
                # Check if gapA was created by starting a new gap
                if gapA[current_i][current_j] == dp[current_i-1][current_j] + gap_open:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[current_j-1])
                    current_i -= 1
                elif gapB[current_i][current_j] == dp[current_i][current_j-1] + gap_open:
                    aligned_a.append(seq_a[current_i-1])
                    aligned_b.append('-')
                    current_j -= 1
                else:
                    # Handle error case
                    pass

        # Reverse the aligned sequences
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))

        return dp[len_a][len_b], aligned_a, aligned_b
    else: # local mode
        # Initialize DP tables
        dp = [[0]*(len_b+1) for _ in range(len_a+1)]
        gapA = [[0]*(len_b+1) for _ in range(len_a+1)]
        gapB = [[0]*(len_b+1) for _ in range(len_a+1)]

        # Fill the DP tables
        for i in range(1, len_a+1):
            for j in range(1, len_b+1):
                # Compute score for match/mismatch
                score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                option1 = dp[i-1][j-1] + score
                option2 = gapA[i-1][j] + gap_extend
                option3 = gapB[i][j-1] + gap_extend
                option4 = 0
                dp[i][j] = max(option1, option2, option3, option4)

                # Compute gapA[i][j]
                option1_gapA = dp[i-1][j] + gap_open
                option2_gapA = gapA[i-1][j] + gap_extend
                gapA[i][j] = max(option1_gapA, option2_gapA, 0)

                # Compute gapB[i][j]
                option1_gapB = dp[i][j-1] + gap_open
                option2_gapB = gapB[i][j-1] + gap_extend
                gapB[i][j] = max(option1_gapB, option2_gapB, 0)

        # Find the maximum value in dp
        max_score = max(max(row) for row in dp)
        if max_score == 0:
            return (0, "", "")

        # Find the position of the maximum score
        for i in range(len_a+1):
            for j in range(len_b+1):
                if dp[i][j] == max_score:
                    current_i, current_j = i, j
                    break
            else:
                continue
            break

        # Backtrack from current_i and current_j
        aligned_a = []
        aligned_b = []

        while current_i > 0 and current_j > 0:
            # Check if current cell was a match/mismatch
            if dp[current_i][current_j] == dp[current_i-1][current_j-1] + (match if seq_a[current_i-1] == seq_b[current_j-1] else mismatch):
                aligned_a.append(seq_a[current_i-1])
                aligned_b.append(seq_b[current_j-1])
                current_i -= 1
                current_j -= 1
            elif gapA[current_i][current_j] == gapA[current_i-1][current_j] + gap_extend:
                aligned_a.append('-')
                aligned_b.append(seq_b[current_j-1])
                current_i -= 1
            elif gapB[current_i][current_j] == gapB[current_i][current_j-1] + gap_extend:
                aligned_a.append(seq_a[current_i-1])
                aligned_b.append('-')
                current_j -= 1
            else:
                # Check if gapA was created by starting a new gap
                if gapA[current_i][current_j] == dp[current_i-1][current_j] + gap_open:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[current_j-1])
                    current_i -= 1
                elif gapB[current_i][current_j] == dp[current_i][current_j-1] + gap_open:
                    aligned_a.append(seq_a[current_i-1])
                    aligned_b.append('-')
                    current_j -= 1
                else:
                    # Handle error case
                    pass

        # Reverse the aligned sequences
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))

        return max_score, aligned_a, aligned_b

