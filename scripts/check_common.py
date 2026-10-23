"""Check (or with --fix, restore) that every task's tests/sandbox_worker.py matches the
canonical harness/common/sandbox_worker.py."""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "harness" / "common" / "sandbox_worker.py"


def main():
    fix = "--fix" in sys.argv
    stale = []
    for task in sorted((ROOT / "tasks").iterdir()):
        copy = task / "tests" / "sandbox_worker.py"
        if not copy.exists() or copy.read_bytes() != CANONICAL.read_bytes():
            stale.append(task.name)
            if fix:
                shutil.copy(CANONICAL, copy)
    print("stale copies:", stale or "none", "(fixed)" if fix and stale else "")
    return 1 if stale and not fix else 0


if __name__ == "__main__":
    sys.exit(main())
