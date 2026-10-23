import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    len_a, len_b = len(seq_a), len(seq_b)
    
    if mode == "global":
        main = np.full((len_a + 1, len_b + 1), -np.inf, dtype=np.int64)
        gapA = np.full((len_a + 1, len_b + 1), -np.inf, dtype=np.int64)
        gapB = np.full((len_a + 1, len_b + 1), -np.inf, dtype=np.int64)
        main[0, 0] = 0
        
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if i == 0 and j == 0:
                    continue
                
                # Compute main[i][j]
                if i > 0 and j > 0:
                    score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                    main[i, j] = max(
                        main[i-1, j-1] + score,
                        gapA[i-1, j] + gap_extend,
                        gapB[i, j-1] + gap_extend
                    )
                
                # Compute gapA[i][j]
                if i > 0:
                    gapA[i, j] = max(
                        gapA[i-1, j] + gap_extend,
                        main[i-1, j] + gap_open
                    )
                
                # Compute gapB[i][j]
                if j > 0:
                    gapB[i, j] = max(
                        gapB[i, j-1] + gap_extend,
                        main[i, j-1] + gap_open
                    )
        
        score = main[len_a, len_b]
        
        # Backtrack
        aligned_a = []
        aligned_b = []
        i, j = len_a, len_b
        
        while i > 0 and j > 0:
            # Check main
            if main[i, j] == main[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif gapA[i, j] == gapA[i-1, j] + gap_extend:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif gapB[i, j] == gapB[i, j-1] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                j -= 1
            else:
                # Check gapA start
                if gapA[i, j] == main[i-1, j] + gap_open:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                # Check gapB start
                elif gapB[i, j] == main[i, j-1] + gap_open:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    j -= 1
                else:
                    break
        
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
        return (score, aligned_a, aligned_b)
    else:  # local mode
        main = np.full((len_a + 1, len_b + 1), 0, dtype=np.int64)
        gapA = np.full((len_a + 1, len_b + 1), 0, dtype=np.int64)
        gapB = np.full((len_a + 1, len_b + 1), 0, dtype=np.int64)
        
        for i in range(len_a + 1):
            for j in range(len_b + 1):
                if i == 0 and j == 0:
                    continue
                
                # Compute main[i][j]
                if i > 0 and j > 0:
                    score = match if seq_a[i-1] == seq_b[j-1] else mismatch
                    main[i, j] = max(
                        main[i-1, j-1] + score,
                        gapA[i-1, j] + gap_extend,
                        gapB[i, j-1] + gap_extend
                    )
                
                # Compute gapA[i][j]
                if i > 0:
                    gapA[i, j] = max(
                        gapA[i-1, j] + gap_extend,
                        main[i-1, j] + gap_open
                    )
                
                # Compute gapB[i][j]
                if j > 0:
                    gapB[i, j] = max(
                        gapB[i, j-1] + gap_extend,
                        main[i, j-1] + gap_open
                    )
        
        max_score = main.max()
        if max_score <= 0:
            return (0, "", "")
        
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(main), main.shape)
        
        # Backtrack
        aligned_a = []
        aligned_b = []
        while i > 0 and j > 0:
            # Check main
            if main[i, j] == main[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch):
                aligned_a.append(seq_a[i-1])
                aligned_b.append(seq_b[j-1])
                i -= 1
                j -= 1
            elif gapA[i, j] == gapA[i-1, j] + gap_extend:
                aligned_a.append('-')
                aligned_b.append(seq_b[j-1])
                i -= 1
            elif gapB[i, j] == gapB[i, j-1] + gap_extend:
                aligned_a.append(seq_a[i-1])
                aligned_b.append('-')
                j -= 1
            else:
                # Check gapA start
                if gapA[i, j] == main[i-1, j] + gap_open:
                    aligned_a.append('-')
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                # Check gapB start
                elif gapB[i, j] == main[i, j-1] + gap_open:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append('-')
                    j -= 1
                else:
                    break
        
        aligned_a = ''.join(reversed(aligned_a))
        aligned_b = ''.join(reversed(aligned_b))
        return (max_score, aligned_a, aligned_b)


