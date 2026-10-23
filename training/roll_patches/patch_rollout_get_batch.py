"""Patch ROLL's GroupQueueManager.get_batch to drop the per-round barrier (idempotent).

get_batch waits on one pending get() per env group and only issues new gets once every group
has delivered (or the batch is full); the leftover gets carry over to the next step. One episode
that runs for the full 30-minute budget therefore stalls batch collection for all groups even
though the other envs finished long before (observed: rollout time per step 80 s -> 394 s ->
1376 s while episodes took 43 s median). The patch re-issues a group's get() as soon as that
group delivered an episode, so fast envs keep feeding the batch. Episodes older than
async_generation_ratio steps are still dropped as before.

Usage: python training/roll_patches/patch_rollout_get_batch.py <roll-dir>
"""

import sys
from pathlib import Path

MARKER = "# biocompute-arb: re-arm group get"
ANCHOR = "                        if current_step - group.create_step > self.async_generation_ratio:\n"
INSERT = (
    f"                        {MARKER} (no per-round barrier: a slow episode must not stall the batch)\n"
    "                        pending.add(asyncio.create_task(self.group_queue[group.group_id].get(),\n"
    "                                                        name=d.get_name()))\n"
)


def main():
    path = Path(sys.argv[1]) / "roll" / "distributed" / "scheduler" / "rollout_scheduler.py"
    src = path.read_text()
    if MARKER in src:
        print(f"already patched: {path}")
        return
    if src.count(ANCHOR) != 1:
        raise SystemExit(f"get_batch in {path} differs from the expected original")
    path.write_text(src.replace(ANCHOR, INSERT + ANCHOR))
    print(f"patched: {path}")


if __name__ == "__main__":
    main()
