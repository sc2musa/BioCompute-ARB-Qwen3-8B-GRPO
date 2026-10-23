from align import align

def test_align():
    # Example 0
    assert align("TATTGCTTTACAAAATTAAGGCCCAAAAGC",
                 "TAATACTTTACAAAGCTACGGCCCAAAAGC",
                 mode="global",
                 match=2, mismatch=-3,
                 gap_open=-5, gap_extend=-2) == (35, "TATTGCTTTACAAAATTAAGGCCCAAAAGC",
                                                  "TAATACTTTACAAAGCTACGGCCCAAAAGC")

    # Example 1
    assert align("AGTGCCGACGGACCAGAACAATGTGGGGAA",
                 "GGTGAAGACGTACCAGAACAATATGTGGATC",
                 mode="local",
                 match=1, mismatch=-1,
                 gap_open=-2, gap_extend=-2) == (18, "AGTGCCGACGGACCAGAACAATGTGGGGAA",
                                                  "GGTGAAGACGTACCAGAACAATATGTGGATC")

    # Example 2
    assert align("TGATGGACTCGGATCGTGTCGTAAGACGAG",
                 "TGATGGACTCGGGTCGTTTCGTAAGACTATAGAG",
                 mode="global",
                 match=5, mismatch=-4,
                 gap_open=-10, gap_extend=-1) == (119, "TGATGGACTCGGATCGTGTCGTAAGACGAG",
                                                  "TGATGGACTCGGGTCGTTTCGTAAGACTATAGAG")

    # Example 3
    assert align("TCAATCTGGAACCTCGCTGTTGGTAAGCCA",
                 "TCTTTCTGGAACCTGGACAGTTGGTTAGCCA",
                 mode="local",
                 match=3, mismatch=-2,
                 gap_open=-4, gap_extend=-3) == (61, "TCAATCTGGAACCTCGCTGTTGGTAAGCCA",
                                                  "TCTTTCTGGAACCTGGACAGTTGGTTAGCCA")

if __name__ == "__main__":
    test_align()