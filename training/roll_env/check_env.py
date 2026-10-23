"""Self-test of the ROLL BioCompute environment without a model (run on the server as root).

Plays scripted replies through the env as ROLL would (gem.make("biocompute"), raw model text
with <|im_end|>): a reply without a tool call, an unparsable call, a normal bash call, writing
the instance's reference solution, submit. Checks the feedback texts, the turn prefix, that the
oracle episode gets reward 1 + solve_bonus and an empty one gets 0, and prints the prompt the
env manager would render for the next turn. The reference solution of a *training* instance is
only used here to test grading; it never enters training data.

Usage: python training/roll_env/check_env.py --roll-dir <roll> --instance <train-task-dir> \
           --tokenizer <model-dir> [--sandbox-user agent]
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
END = "<|im_end|>"


def call(name, **arguments):
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>{END}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll-dir", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--sandbox-user", default="agent")
    args = ap.parse_args()
    sys.path[:0] = [args.roll_dir, str(REPO_DIR)]
    import gem
    import roll.pipeline.agentic.env  # noqa: F401  (registers "biocompute")
    from harness import agent
    from transformers import AutoTokenizer

    instance = Path(args.instance)
    work = Path(tempfile.mkdtemp(prefix="biocompute_check_"))
    work.chmod(0o711)  # mkdtemp is 0700; the sandbox user must reach its workspace below
    task_list = work / "tasks.txt"
    task_list.write_text(instance.name + "\n")
    env = gem.make("biocompute", repo_dir=str(REPO_DIR), tasks_root=str(instance.parent),
                   work_root=str(work / "env"), task_list=str(task_list),
                   sandbox_user=args.sandbox_user, episode_log=str(work / "episodes.jsonl"))
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    obs, info = env.reset(seed=7)
    assert [m["role"] for m in obs] == ["system", "user"] and len(info["tools"]) == 4
    obs, r, done, _, _ = env.step("I will now write the simulator." + END)
    assert not done and r == 0 and obs[-1]["content"].startswith(agent.CONTINUE_PROMPT[:30]), obs[-1]
    obs, r, done, _, _ = env.step('<tool_call>\n{"name": "bash", "arguments": {bad json}}\n</tool_call>' + END)
    assert not done and "could not be parsed" in obs[-1]["content"], obs[-1]
    obs, r, done, _, _ = env.step(call("bash", command="ls -la; cat /etc/hostname >/dev/null"))
    assert not done and obs[-1]["role"] == "tool" and obs[-1]["content"].startswith("[turn 3/30"), obs[-1]
    print("tool output:", obs[-1]["content"][:200].replace("\n", " | "))
    obs, r, done, _, _ = env.step(call("bash", command=f"ls {instance}/tests"))
    print("hidden tests visible to agent?:", obs[-1]["content"][:160].replace("\n", " | "))
    solution = (instance / "solution" / "simulate.py").read_text()
    obs, r, done, _, _ = env.step("Writing the solver.\n\n" + call("write_file", path="simulate.py", content=solution))
    obs, r, done, _, _ = env.step(call("bash", command="python check.py", timeout=300))
    print("check.py:", obs[-1]["content"][-300:].replace("\n", " | "))
    for i in range(6):  # more than KEEP_RECENT tool calls: old outputs and arguments get elided
        obs, r, done, _, _ = env.step(call("bash", command=f"echo step{i}"))
    assert not done
    elided = [m for m in obs if m["role"] == "assistant" and m.get("tool_calls")
              and "chars elided" in m["tool_calls"][0]["function"]["arguments"]]
    assert elided, "the old write_file call should be elided"
    prompt = tok.apply_chat_template(obs, tools=info["tools"], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    print(f"--- next prompt ({len(tok(prompt).input_ids)} tokens), tail ---\n{prompt[-1500:]}\n---")
    obs, r, done, _, info2 = env.step(call("submit"))
    assert done, "submit must end the episode"
    print("oracle episode: reward", r, info2["metrics"])
    assert abs(r - 1.5) < 1e-9, r

    env.reset(seed=8)
    _, r, done, _, info3 = env.step(call("submit"))
    print("empty episode: reward", r, info3["metrics"])
    assert done and r == 0.0
    print("episode log:", (work / "episodes.jsonl").read_text().strip().replace("\n", "\n  "))
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
