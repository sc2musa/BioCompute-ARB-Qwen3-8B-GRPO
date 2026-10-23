import numpy as np


def align(seq_a, seq_b, mode="global", match=2, mismatch=-3, gap_open=-5, gap_extend=-2):
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize DP table
    len_a, len_b = len(seq_a), len(seq_b)
    dp = np.zeros((len_a + 1, len_b + 1), dtype=int)
    
    # Initialize first row and column with gap penalties
    for i in range(1, len_a + 1):
        dp[i, 0] = gap_open + (i-1)*gap_extend
    for j in range(1, len_b + 1):
        dp[0, j] = gap_open + (j-1)*gap_extend
    
    # Fill DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate scores for all possible operations
            match_score = dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            delete_score = dp[i-1, j] + gap_extend
            insert_score = dp[i, j-1] + gap_extend
            
            # Choose the maximum score
            dp[i, j] = max(match_score, delete_score, insert_score)
    
    # Handle local mode
    if mode == "local":
        # Find the maximum score in the DP table
        max_score = np.max(dp)
        # Find the position of the maximum score
        i, j = np.unravel_index(np.argmax(dp), dp.shape)
        # Initialize the DP table with -infinity for the traceback
        dp = np.full((len_a + 1, len_b + 1), -np.inf, dtype=int)
        # Initialize the DP table with the maximum score
        dp[i, j] = max_score
        # Fill the DP table again
        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                # Calculate scores for all possible operations
                match_score = dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
                delete_score = dp[i-1, j] + gap_extend
                insert_score = dp[i, j-1] + gap_extend
                
                # Choose the maximum score
                dp[i, j] = max(match_score, delete_score, insert_score)
        # Traceback to find the optimal alignment
        i, j = len_a, len_b
        aligned_a, aligned_b = [], []
        
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                if seq_a[i-1] == seq_b[j-1]:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                else:
                    # Choose the operation with the highest score
                    if dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch) == dp[i, j]:
                        aligned_a.append(seq_a[i-1])
                        aligned_b.append(seq_b[j-1])
                        i -= 1
                        j -= 1
                    elif dp[i-1, j] + gap_extend == dp[i, j]:
                        aligned_a.append(seq_a[i-1])
                        aligned_b.append("-")
                        i -= 1
                    else:
                        aligned_a.append("-")
                        aligned_b.append(seq_b[j-1])
                        j -= 1
            elif i > 0:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            else:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
        
        # Reverse the alignments
        aligned_a = aligned_a[::-1]
        aligned_b = aligned_b[::-1]
        
        # Return the score and alignments
        return max_score, "".join(aligned_a), "".join(aligned_b)
    
    # Handle global mode
    else:
        # Traceback to find the optimal alignment
        i, j = len_a, len_b
        aligned_a, aligned_b = [], []
        
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                if seq_a[i-1] == seq_b[j-1]:
                    aligned_a.append(seq_a[i-1])
                    aligned_b.append(seq_b[j-1])
                    i -= 1
                    j -= 1
                else:
                    # Choose the operation with the highest score
                    if dp[i-1, j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch) == dp[i, j]:
                        aligned_a.append(seq_a[i-1])
                        aligned_b.append(seq_b[j-1])
                        i -= 1
                        j -= 1
                    elif dp[i-1, j] + gap_extend == dp[i, j]:
                        aligned_a.append(seq_a[i-1])
                        aligned_b.append("-")
                        i -= 1
                    else:
                        aligned_a.append("-")
                        aligned_b.append(seq_b[j-1])
                        j -= 1
            elif i > 0:
                aligned_a.append(seq_a[i-1])
                aligned_b.append("-")
                i -= 1
            else:
                aligned_a.append("-")
                aligned_b.append(seq_b[j-1])
                j -= 1
        
        # Reverse the alignments
        aligned_a = aligned_a[::-1]
        aligned_b = aligned_b[::-1]
        
        # Return the score and alignments
        return dp[len_a, len_b], "".join(aligned_a), "".join(aligned_b)