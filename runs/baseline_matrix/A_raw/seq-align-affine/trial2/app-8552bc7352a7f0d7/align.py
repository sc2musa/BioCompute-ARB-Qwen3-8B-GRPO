import numpy as np

def align(seq_a: str, seq_b: str, mode: str = "global",
          match: int = 2, mismatch: int = -3,
          gap_open: int = -5, gap_extend: int = -2) -> tuple[int, str, str]:
    """Return (score, aligned_a, aligned_b)."""
    
    # Initialize the scoring matrix
    m, n = len(seq_a), len(seq_b)
    score_matrix = np.zeros((m + 1, n + 1))
    
    # Initialize the first row and column
    for i in range(1, m + 1):
        score_matrix[i][0] = score_matrix[i - 1][0] + gap_open
    for j in range(1, n + 1):
        score_matrix[0][j] = score_matrix[0][j - 1] + gap_open
    
    # Fill the scoring matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Match/mismatch score
            match_score = score_matrix[i - 1][j - 1] + (match if seq_a[i - 1] == seq_b[j - 1] else mismatch)
            
            # Gap in seq_a
            gap_a_score = score_matrix[i - 1][j] + gap_open
            
            # Gap in seq_b
            gap_b_score = score_matrix[i][j - 1] + gap_open
            
            # Choose the maximum score
            score_matrix[i][j] = max(match_score, gap_a_score, gap_b_score)
    
    # Backtrack to find the optimal alignment
    i, j = m, n
    aligned_a, aligned_b = ",", ","
    
    while i > 0 and j > 0:
        if seq_a[i - 1] == seq_b[j - 1]:
            aligned_a = seq_a[i - 1] + aligned_a
            aligned_b = seq_b[j - 1] + aligned_b
            i -= 1
            j -= 1
        elif score_matrix[i][j] == score_matrix[i - 1][j] + gap_open:
            aligned_a = seq_a[i - 1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        elif score_matrix[i][j] == score_matrix[i][j - 1] + gap_open:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j - 1] + aligned_b
            j -= 1
        else:
            aligned_a = seq_a[i - 1] + aligned_a
            aligned_b = seq_b[j - 1] + aligned_b
            i -= 1
            j -= 1
    
    # Handle remaining characters
    while i > 0:
        aligned_a = seq_a[i - 1] + aligned_a
        aligned_b = "-" + aligned_b
        i -= 1
    while j > 0:
        aligned_a = "-" + aligned_a
        aligned_b = seq_b[j - 1] + aligned_b
        j -= 1
    
    # Calculate the score
    score = 0
    i, j = 0, 0
    while i < len(aligned_a) and j < len(aligned_b):
        if aligned_a[i] == "-" and aligned_b[j] == "-":
            i += 1
            j += 1
            continue
        elif aligned_a[i] == "-":
            # Gap in seq_a
            score += gap_open
            k = j + 1
            while k < len(aligned_b) and aligned_b[k] == "-":
                score += gap_extend
                k += 1
            j = k
        elif aligned_b[j] == "-":
            # Gap in seq_b
            score += gap_open
            k = i + 1
            while k < len(aligned_a) and aligned_a[k] == "-":
                score += gap_extend
                k += 1
            i = k
        else:
            if aligned_a[i] == aligned_b[j]:
                score += match
            else:
                score += mismatch
            i += 1
            j += 1
    
    return (score, aligned_a, aligned_b)

# Test the align function
if __name__ == "__main__":
    seq_a = 'TATTGCTTTACAAAATTAAGGCCCAAAAGC'
    seq_b = 'TAATACTTTACAAAGCTACGGCCCAAAAGC'
    score, aligned_a, aligned_b = align(seq_a, seq_b, mode='global', match=2, mismatch=-3, gap_open=-5, gap_extend=-2)
    print(f"Score: {score}")
    print(f"Aligned A: {aligned_a}")
    print(f"Aligned B: {aligned_b}")