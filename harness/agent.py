"""Minimal tool-calling research agent for an OpenAI-compatible endpoint (e.g. vLLM).

The agent sees only the task instruction and its workspace. Tools: bash, write_file,
read_file, submit. Every request/response is appended to a JSONL trajectory. Older tool
outputs are elided from the prompt (the log keeps them) so long episodes fit the context.
"""

import json
import time

from openai import BadRequestError, OpenAI

SYSTEM_PROMPT = """You are an autonomous research engineer working in a Linux terminal.
Your current directory is the task workspace; all your files live there.
Work step by step: inspect the workspace, write code, run experiments, check your results
against the public examples, and fix problems. Tools run real commands; nothing is
simulated. You cannot install packages or access the internet.
When your solution is finished and verified, call the `submit` tool."""

TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a bash command in the workspace and return exit code and output.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The bash command to run."},
            "timeout": {"type": "integer", "description": "Seconds before the command is killed (default 120, max 600)."},
        }, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a text file in the workspace.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace."},
            "content": {"type": "string", "description": "Full file content."},
        }, "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file from the workspace.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace."},
        }, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "submit",
        "description": "Finish the task. Call only when the solution is complete.",
        "parameters": {"type": "object", "properties": {}}}},
]

BUDGET_PROMPT = """
You have at most {turns} turns (one model reply each) and {minutes} minutes in total. Every tool
result starts with the current turn and the time left. Plan your work to fit, and call `submit`
before the budget runs out."""
MAX_COMMAND_TIMEOUT = 600
REPEAT_LIMIT = 3     # same tool call this many times within the window triggers the note
REPEAT_WINDOW = 8    # most recent tool calls considered by the repeat guard
REPEAT_NOTE = ("\n\n[harness note] You have issued this exact tool call {n} times among your "
               "last {w} calls without changing any file in between. Repeating it will not help: "
               "inspect or change your code, try a different test, or call `submit` if you are done.")
# Ignoring this many consecutive repeat notes ends the episode (the workspace is still graded)
LOOP_STOP_NOTES = 5
# Remind the agent to wrap up when this few turns or seconds are left
LOW_BUDGET_TURNS = 5
LOW_BUDGET_SEC = 300
LOW_BUDGET_NOTE = ("\n\n[harness note] Only {turns} turns / {sec}s left. Make sure your best "
                   "working solution is saved, then call `submit`.")
# Context management: the model only sees the latest tool outputs / file contents in full.
KEEP_RECENT = 6
KEEP_RECENT_ON_OVERFLOW = 2
ELIDE_MIN_CHARS = 400
ELIDED_OUTPUT = "[older tool output ({n} chars) elided to save context; re-run the tool if needed]"
ELIDED_PROSE = "[earlier message ({n} chars) elided to save context]"
# (keep_recent, elide_prose) per attempt; the second one is used after a context-length error
VIEW_DEFAULT = (KEEP_RECENT, False)
VIEW_ON_OVERFLOW = (KEEP_RECENT_ON_OVERFLOW, True)

# Feedback when a reply has no executable tool call (the model must learn why)
CONTINUE_PROMPT = ("Your reply contained no tool call, so nothing was done. Act now: call a tool "
                   "(write_file, bash, read_file), or `submit` when done.")
TEXT_REPEAT_NOTE = ("\n\n[harness note] You have sent this same message {n} times without calling a "
                    "tool. Plans are not executed: call write_file or bash now.")
TRUNCATED_PROMPT = ("Your previous reply hit the {max_tokens}-token output limit and was cut off, so no "
                    "tool was run. Keep each reply shorter: write smaller files (or split the code "
                    "into several files) and avoid repeating text.")
BAD_CALL_PROMPT = ("Your tool call could not be parsed, so nothing was run: {error}. Tool arguments "
                   "must be valid JSON: escape backslashes (\\\\) and newlines (\\n) inside strings, "
                   "or use write_file for code.")
CODE_IN_TEXT_PROMPT = ("Code written in a message is not saved or executed. Use write_file to save "
                       "it and bash to run it. Call `submit` when done.")


def _elide_arguments(arguments):
    """Shorten long string arguments (e.g. write_file content) of an old tool call."""
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return arguments
    for key, value in args.items():
        if isinstance(value, str) and len(value) > ELIDE_MIN_CHARS:
            args[key] = f"<{len(value)} chars elided>"
    return json.dumps(args)


def compact_view(messages, keep_recent, elide_prose=False):
    """Copy of the conversation with all but the last `keep_recent` tool results and tool-call
    arguments elided; with elide_prose, also the text of assistant messages before them.
    The full history is still written to the trajectory log."""
    tool_idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    call_idx = [i for i, m in enumerate(messages) if m.get("tool_calls")]
    old_tools = set(tool_idx[:-keep_recent] if keep_recent else tool_idx)
    old_calls = set(call_idx[:-keep_recent] if keep_recent else call_idx)
    recent_start = min((set(call_idx) - old_calls) | {len(messages)})
    view = []
    for i, m in enumerate(messages):
        if i in old_tools and len(m["content"]) > ELIDE_MIN_CHARS:
            m = {**m, "content": ELIDED_OUTPUT.format(n=len(m["content"]))}
        elif i in old_calls:
            calls = [{**c, "function": {**c["function"],
                                        "arguments": _elide_arguments(c["function"]["arguments"])}}
                     for c in m["tool_calls"]]
            m = {**m, "tool_calls": calls}
        if (elide_prose and m["role"] == "assistant" and i < recent_start
                and len(m["content"] or "") > ELIDE_MIN_CHARS):
            m = {**m, "content": ELIDED_PROSE.format(n=len(m["content"]))}
        view.append(m)
    return view


def no_tool_feedback(content, finish_reason, max_tokens):
    """Message sent back when a reply contained no executable tool call."""
    if finish_reason == "length":
        return TRUNCATED_PROMPT.format(max_tokens=max_tokens)
    if "<tool_call>" in content:
        block = content.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
        try:
            json.loads(block)
            error = "the call was not in the expected format"
        except json.JSONDecodeError as exc:
            error = f"invalid JSON ({exc.msg} at character {exc.pos})"
        return BAD_CALL_PROMPT.format(error=error)
    if "```" in content:
        return CODE_IN_TEXT_PROMPT
    return CONTINUE_PROMPT


class AgentConfig:
    def __init__(self, base_url, model, api_key="EMPTY", max_turns=30, temperature=0.7,
                 top_p=0.8, max_tokens=4096, enable_thinking=False, seed=None,
                 time_budget_sec=1800, repeat_guard=False, presence_penalty=0.0):
        self.base_url, self.model, self.api_key = base_url, model, api_key
        self.max_turns, self.temperature, self.top_p = max_turns, temperature, top_p
        self.max_tokens, self.enable_thinking, self.seed = max_tokens, enable_thinking, seed
        self.time_budget_sec, self.repeat_guard = time_budget_sec, repeat_guard
        self.presence_penalty = presence_penalty


def _call_tool(sandbox, name, args, max_timeout=MAX_COMMAND_TIMEOUT):
    if name == "bash":
        timeout = max(1, min(int(args.get("timeout") or 120), MAX_COMMAND_TIMEOUT, max_timeout))
        return sandbox.run(args["command"], timeout)
    if name == "write_file":
        return sandbox.write_file(args["path"], args["content"])
    if name == "read_file":
        return sandbox.read_file(args["path"])
    return f"unknown tool: {name}"


class _ToolRunner:
    """Executes tool calls with the repeat guard and the visible budget."""

    def __init__(self, sandbox, config, stats, start):
        self.sandbox, self.config, self.stats, self.start = sandbox, config, stats, start
        self.recent_calls, self.ignored_notes = [], 0

    def seconds_left(self):
        return max(0, int(self.config.time_budget_sec - (time.time() - self.start)))

    def run(self, tc, turn):
        """Return the observation for one tool call (turn is 1-based)."""
        self.stats["tool_calls"] += 1
        name = tc.function.name
        # A repeat is the same call on an unchanged workspace: re-running check.py after an
        # edit is exactly what the instruction asks for and must not be flagged
        signature = (name, tc.function.arguments, self.sandbox.fingerprint())
        try:
            args = json.loads(tc.function.arguments or "{}")
            # A command may not outlive the episode's time budget
            output = "submitted" if name == "submit" else _call_tool(
                self.sandbox, name, args, max_timeout=self.seconds_left())
        except Exception as exc:
            self.stats["tool_errors"] += 1
            output = f"tool error: {type(exc).__name__}: {exc}"
        self.recent_calls = (self.recent_calls + [signature])[-REPEAT_WINDOW:]
        count = self.recent_calls.count(signature)
        if count > 1:
            self.stats["repeated_calls"] += 1
        if self.config.repeat_guard and count >= REPEAT_LIMIT:
            self.stats["repeat_notes"] += 1
            self.ignored_notes += 1
            output += REPEAT_NOTE.format(n=count, w=REPEAT_WINDOW)
        else:
            self.ignored_notes = 0
        turns_left, sec_left = self.config.max_turns - turn, self.seconds_left()
        if name != "submit" and (turns_left <= LOW_BUDGET_TURNS or sec_left <= LOW_BUDGET_SEC):
            output += LOW_BUDGET_NOTE.format(turns=turns_left, sec=sec_left)
        return f"[turn {turn}/{self.config.max_turns}, {sec_left}s left]\n{output}"

    def text_reply(self, content):
        """Repeat guard for replies without a tool call; return the note to append (or "")."""
        self.recent_calls = (self.recent_calls + [("<text>", content)])[-REPEAT_WINDOW:]
        count = self.recent_calls.count(("<text>", content))
        if self.config.repeat_guard and count >= REPEAT_LIMIT:
            self.stats["repeat_notes"] += 1
            self.ignored_notes += 1
            return TEXT_REPEAT_NOTE.format(n=count)
        self.ignored_notes = 0
        return ""

    def looping(self):
        return self.ignored_notes >= LOOP_STOP_NOTES


def run_agent(instruction, sandbox, config, trajectory_path):
    """Run one episode; return summary statistics."""
    client = OpenAI(base_url=config.base_url, api_key=config.api_key, timeout=600)
    system = SYSTEM_PROMPT + BUDGET_PROMPT.format(turns=config.max_turns,
                                                  minutes=round(config.time_budget_sec / 60))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": instruction}]
    stats = {"turns": 0, "tool_calls": 0, "tool_errors": 0, "prompt_tokens": 0,
             "completion_tokens": 0, "submitted": False, "stop_reason": "max_turns",
             "repeated_calls": 0, "repeat_notes": 0, "overflow_retries": 0}
    start = time.time()
    tools = _ToolRunner(sandbox, config, stats, start)

    def request(view):
        return client.chat.completions.create(
            model=config.model, messages=compact_view(messages, *view), tools=TOOLS,
            tool_choice="auto", temperature=config.temperature, top_p=config.top_p,
            max_tokens=config.max_tokens, seed=config.seed, presence_penalty=config.presence_penalty,
            extra_body={"chat_template_kwargs": {"enable_thinking": config.enable_thinking}},
        )

    with open(trajectory_path, "w", encoding="utf-8") as log:
        def record(event):
            log.write(json.dumps({"t": round(time.time() - start, 2), **event}, ensure_ascii=False) + "\n")
            log.flush()

        record({"type": "start", "messages": messages})
        for turn in range(config.max_turns):
            if time.time() - start > config.time_budget_sec:
                stats["stop_reason"] = "time_budget"
                break
            view = VIEW_DEFAULT
            try:
                try:
                    resp = request(view)
                except BadRequestError as exc:
                    if "context length" not in str(exc):
                        raise
                    stats["overflow_retries"] += 1
                    view = VIEW_ON_OVERFLOW
                    resp = request(view)
            except Exception as exc:  # persistent overflow or server error ends the episode
                stats["stop_reason"] = f"api_error: {type(exc).__name__}: {str(exc)[:300]}"
                record({"type": "api_error", "error": stats["stop_reason"]})
                break
            stats["turns"] = turn + 1
            if resp.usage:
                stats["prompt_tokens"] += resp.usage.prompt_tokens
                stats["completion_tokens"] += resp.usage.completion_tokens
            msg = resp.choices[0].message
            assistant = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
            messages.append(assistant)
            # `view` is how the prompt was compacted for this reply (needed to rebuild it for SFT)
            record({"type": "assistant", "message": assistant,
                    "reasoning": getattr(msg, "reasoning_content", None),
                    "finish_reason": resp.choices[0].finish_reason, "view": list(view)})
            if not msg.tool_calls:
                feedback = no_tool_feedback(assistant["content"], resp.choices[0].finish_reason,
                                            config.max_tokens) + tools.text_reply(assistant["content"])
                messages.append({"role": "user", "content": feedback})
                record({"type": "user", "content": feedback})
                if tools.looping():
                    stats["stop_reason"] = "repeat_loop"
                    break
                continue
            for tc in msg.tool_calls:
                output = tools.run(tc, turn + 1)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
                record({"type": "tool", "name": tc.function.name,
                        "arguments": tc.function.arguments, "output": output})
                if tc.function.name == "submit":
                    stats["submitted"] = True
            if stats["submitted"]:
                stats["stop_reason"] = "submitted"
                break
            if tools.looping():
                stats["stop_reason"] = "repeat_loop"
                break
        stats["wall_seconds"] = round(time.time() - start, 1)
        record({"type": "end", "stats": stats})
    return stats
