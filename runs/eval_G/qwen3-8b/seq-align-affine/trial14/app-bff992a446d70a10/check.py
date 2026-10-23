import json
from align import align

with open("$ROOT/runs/eval_round4/base_G/seq-align-affine/trial14/app-bff992a446d70a10/examples.json") as f:
    examples = json.load(f)

for ex in examples:
    seq_a = ex["a"]
    seq_b = ex["b"]
    mode = ex["mode"]
    expected_score = ex["expected_score"]
    match = ex.get("match", 2)
    mismatch = ex.get("mismatch", -3)
    gap_open = ex.get("gap_open", -5)
    gap_extend = ex.get("gap_extend", -2)
    score, _, _ = align(seq_a, seq_b, mode, match, mismatch, gap_open, gap_extend)
    assert score == expected_score, f"Failed for {ex}"

print("All examples passed!")