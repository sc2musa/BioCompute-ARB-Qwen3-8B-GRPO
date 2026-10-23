"""Turn sampled agent trajectories into SFT samples (rejection-sampling fine-tuning).

Usage (server, roll env):
    python training/build_sft_data.py --runs <sampling-run-dir> --tokenizer <model-dir> \
        --out <file.jsonl> [--min-score 0.5]

Selection, per training instance: keep every trajectory with reward 1; if there is none, keep the
single best trajectory when its score >= --min-score and it beats the instance's mean score.
Each assistant turn of a kept trajectory becomes one sample whose prompt is exactly the text the
model saw at that step (Qwen3 chat template, tools, non-thinking, harness context compaction)
and whose response is what it generated. Loss is applied to the response only.

Trajectories sampled with method guidance (`--guidance method`) are trained on with the
goal-only instruction: the appended guidance.md is removed from the first user message, so the
model learns to take the right approach without being told (STaR-style rationalization).
This needs --tasks-root to rebuild the goal-only instruction.

--uncovered-out writes the instances that have no kept trajectory (comma-separated), i.e. the
instances to resample with method guidance.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.agent import TOOLS, VIEW_DEFAULT, compact_view  # noqa: E402
from harness.runner import build_instruction  # noqa: E402

END_TOKEN = "<|im_end|>"
GUIDANCE_REF = re.compile(r"\bguidance\b|\bguideline", re.IGNORECASE)
# The only feedback of logs written before feedback messages were recorded
LEGACY_CONTINUE_PROMPT = "Continue working using the tools. Call `submit` when done."


def goal_instruction(result, trajectory_dir, tasks_root):
    """(instruction the agent saw, goal-only version) for a method-guidance trajectory."""
    task_dir = Path(tasks_root) / result["task"]
    app_dir = trajectory_dir / result["workspace"]
    public_check = result["agent"].get("public_check", False)
    return (build_instruction(task_dir, "method", app_dir, public_check),
            build_instruction(task_dir, "goal", app_dir, public_check))


def rebuild_turns(trajectory_path, instruction_swap=None):
    """Yield (messages_before_turn, assistant_event) for every assistant turn.

    The assistant event carries the message, its finish_reason and the prompt view
    (keep_recent, elide_prose) the model saw. Feedback messages after replies without a tool
    call are taken from the log ("user" events); older logs did not record them and always
    used LEGACY_CONTINUE_PROMPT.
    instruction_swap = (seen, replacement) replaces the first user message, which must equal
    `seen` exactly (otherwise ValueError: the trajectory does not match its task).
    """
    events = [json.loads(line) for line in Path(trajectory_path).open(encoding="utf-8")]
    messages = list(events[0]["messages"])
    if instruction_swap:
        seen, replacement = instruction_swap
        if messages[1]["role"] != "user" or messages[1]["content"] != seen:
            raise ValueError(f"instruction mismatch in {trajectory_path}")
        messages[1] = {"role": "user", "content": replacement}
    logs_feedback = any(e["type"] == "user" for e in events)
    pending = []  # tool_call ids awaiting their tool result, in order
    for e in events[1:]:
        if e["type"] == "assistant":
            msg = e["message"]
            yield list(messages), e
            messages.append(msg)
            pending = [c["id"] for c in msg.get("tool_calls") or []]
            if not pending and not logs_feedback:
                messages.append({"role": "user", "content": LEGACY_CONTINUE_PROMPT})
        elif e["type"] == "user":
            messages.append({"role": "user", "content": e["content"]})
        elif e["type"] == "tool" and pending:
            messages.append({"role": "tool", "tool_call_id": pending.pop(0), "content": e["output"]})


def is_training_target(event):
    """Only well-formed tool calls are learned; cut-off or unparsable replies stay as context."""
    return event.get("finish_reason") != "length" and bool(event["message"].get("tool_calls"))


def mentions_guidance(event):
    """A reply that refers to the method guidance cannot be learned once the guidance is removed
    from the prompt (it would teach the model to cite text it was never shown)."""
    text = (event.get("reasoning") or "") + (event["message"].get("content") or "")
    return bool(GUIDANCE_REF.search(text))


def render(tokenizer, history, event, thinking=False):
    """(prompt, response) exactly as the model saw and produced them.

    In thinking mode the Qwen3 template renders the reasoning only for the turn being generated
    (earlier turns carry their visible content only, as the harness sent them), and the
    generation prompt has no empty <think> block; the response then starts with <think>.
    """
    assistant = event["message"]
    view = compact_view(history, *event.get("view", VIEW_DEFAULT))
    prompt = tokenizer.apply_chat_template(view, tools=TOOLS, add_generation_prompt=True,
                                           enable_thinking=thinking, tokenize=False)
    calls = [{"type": "function", "function": {"name": c["function"]["name"],
                                               "arguments": json.loads(c["function"]["arguments"] or "{}")}}
             for c in assistant.get("tool_calls") or []]
    target = {"role": "assistant", "content": assistant["content"], "tool_calls": calls}
    if thinking:
        target["reasoning_content"] = event.get("reasoning") or ""
    full = tokenizer.apply_chat_template(view + [target], tools=TOOLS, add_generation_prompt=False,
                                         enable_thinking=thinking, tokenize=False)
    if not full.startswith(prompt):
        return None
    response = full[len(prompt):]
    return prompt, response[: response.index(END_TOKEN) + len(END_TOKEN)]


def select(results, min_score):
    by_instance = defaultdict(list)
    for r in results:
        by_instance[r["task"]].append(r)
    kept = []
    for task, rs in by_instance.items():
        winners = [r for r in rs if r["reward"] == 1]
        if not winners:
            best = max(rs, key=lambda r: r["score"])
            mean = sum(r["score"] for r in rs) / len(rs)
            if best["score"] >= min_score and best["score"] > mean:
                winners = [best]
        kept += [r for r in winners if r["agent"].get("submitted")]
    uncovered = sorted(set(by_instance) - {r["task"] for r in kept})
    return kept, len(by_instance), uncovered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, nargs="+")
    ap.add_argument("--tokenizer", default=None, help="required unless only --uncovered-out is wanted")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tasks-root", default=None,
                    help="training task directory; required when a run used method guidance")
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=32768,
                    help="drop samples whose prompt+response exceed the SFT sequence length")
    ap.add_argument("--uncovered-out", default=None,
                    help="write instances without any kept trajectory (comma-separated) here")
    args = ap.parse_args()

    results = []
    for run in args.runs:
        for p in Path(run).resolve().rglob("result.json"):  # absolute, as the agent saw it
            r = json.loads(p.read_text())
            r["_dir"] = p.parent
            results.append(r)
    kept, n_instances, uncovered = select(results, args.min_score)
    if args.uncovered_out:
        Path(args.uncovered_out).write_text(",".join(uncovered))
    per_task, per_source = defaultdict(int), defaultdict(int)
    for r in kept:
        per_task[r["task"].split("__")[0]] += 1
        per_source[r["agent"].get("guidance", "goal")] += 1
    summary = {"instances": n_instances, "trajectories": len(results), "kept": len(kept),
               "kept_per_task": per_task, "kept_per_guidance": per_source,
               "uncovered": len(uncovered)}
    if args.out:
        if not args.tokenizer:
            ap.error("--out needs --tokenizer")
        summary.update(write_samples(kept, args))
    print(json.dumps(summary))


def write_samples(kept, args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    n_samples = n_skipped = n_not_target = n_guidance_ref = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for r in kept:
            guidance = r["agent"].get("guidance", "goal")
            thinking = bool(r["agent"].get("thinking"))
            swap = None
            if guidance == "method":
                if not args.tasks_root:
                    raise SystemExit("method-guidance trajectories need --tasks-root")
                swap = goal_instruction(r, r["_dir"], args.tasks_root)
            for history, event in rebuild_turns(r["_dir"] / "trajectory.jsonl", swap):
                if not is_training_target(event):
                    n_not_target += 1
                    continue
                if guidance == "method" and mentions_guidance(event):
                    n_guidance_ref += 1
                    continue
                rendered = render(tokenizer, history, event, thinking)
                if rendered is None or len(tokenizer(rendered[0] + rendered[1])["input_ids"]) > args.max_tokens:
                    n_skipped += 1
                    continue
                out.write(json.dumps({"prompt_text": rendered[0], "response_text": rendered[1],
                                      "task": r["task"], "score": r["score"],
                                      "sampled_with": guidance, "thinking": thinking},
                                     ensure_ascii=False) + "\n")
                n_samples += 1
    return {"samples": n_samples, "skipped_turns": n_skipped, "not_target_turns": n_not_target,
            "guidance_reference_turns": n_guidance_ref}


if __name__ == "__main__":
    main()
