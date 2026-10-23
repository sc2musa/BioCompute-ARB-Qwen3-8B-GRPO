import json

from align import align

with open("$ROOT/runs/eval_round4/rft4_G/seq-align-affine/trial16/app-850b9a8a24bc5e1c/examples.json") as f:
    examples = json.load(f)

for example in examples:
    seq_a = example["seq_a"]
    seq_b = example["seq_b"]
    mode = example["mode"]
    match = example.get("match", 2)
    mismatch = example.get("mismatch", -3)
    gap_open = example.get("gap_open", -5)
    gap_extend = example.get("gap_extend", -2)
    expected_score = example["expected_score"]
    score, aligned_a, aligned_b = align(seq_a, seq_b, mode, match, mismatch, gap_open, gap_extend)
    assert score == expected_score, f"Failed for {example}"

print("All examples passed!")