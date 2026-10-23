"""Patch ROLL's log_probs_from_logits to a memory-efficient chunked version (idempotent).

The original casts the full [batch, seq, vocab] logits to fp32 and keeps the fp32 log_softmax for
backward: for a 32k-token micro-batch with Qwen3's 152k vocabulary that is ~20 GB each and the
training step ran out of memory. The patched version computes gather(logits) - logsumexp(logits)
on ~4k-token chunks under activation checkpointing, so only one fp32 chunk exists at a time and
it is recomputed in backward. The result is mathematically identical.

Usage: python training/roll_patches/patch_chunked_logprobs.py <roll-dir>
"""

import sys
from pathlib import Path

MARKER = "# biocompute-arb: chunked log_probs_from_logits"
ORIGINAL = '''def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)
'''
PATCHED = f'''{MARKER}
LOG_PROBS_CHUNK_TOKENS = 4096


def _chunk_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    picked = logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return picked - torch.logsumexp(logits, dim=-1)


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """log_softmax(logits)[labels], computed on sequence chunks of ~LOG_PROBS_CHUNK_TOKENS tokens
    (checkpointed when gradients are needed) so the fp32 [*, vocab] tensor never exists in full."""
    rows = max(1, LOG_PROBS_CHUNK_TOKENS // max(1, logits.shape[0]))
    use_ckpt = torch.is_grad_enabled() and logits.requires_grad
    out = []
    for start in range(0, logits.shape[1], rows):
        chunk, chunk_labels = logits[:, start:start + rows], labels[:, start:start + rows]
        if use_ckpt:
            out.append(torch.utils.checkpoint.checkpoint(_chunk_log_probs, chunk, chunk_labels,
                                                         use_reentrant=False))
        else:
            out.append(_chunk_log_probs(chunk, chunk_labels))
    return torch.cat(out, dim=1)
'''


def main():
    path = Path(sys.argv[1]) / "roll" / "utils" / "functionals.py"
    src = path.read_text()
    if MARKER in src:
        print(f"already patched: {path}")
        return
    if src.count(ORIGINAL) != 1:
        raise SystemExit(f"log_probs_from_logits in {path} differs from the expected original")
    if "import torch.utils.checkpoint" not in src:  # after `import torch` (the file starts with __future__)
        if "\nimport torch\n" not in src:
            raise SystemExit(f"no `import torch` line in {path}")
        src = src.replace("\nimport torch\n", "\nimport torch\nimport torch.utils.checkpoint\n", 1)
    path.write_text(src.replace(ORIGINAL, PATCHED))
    print(f"patched: {path}")


if __name__ == "__main__":
    main()
