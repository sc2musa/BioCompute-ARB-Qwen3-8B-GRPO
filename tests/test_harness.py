"""Tests for harness behaviour that the agent interacts with (run: python -m pytest tests/).

The model is replaced by a scripted fake client, so no GPU or network is needed.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import agent as agent_mod  # noqa: E402
from harness.agent import AgentConfig, run_agent  # noqa: E402
from harness.sandbox import Sandbox, run_in_group  # noqa: E402


def _call(i, name, **args):
    fn = SimpleNamespace(name=name, arguments=json.dumps(args))
    return SimpleNamespace(id=f"c{i}", type="function", function=fn,
                           model_dump=lambda fn=fn, i=i: {"id": f"c{i}", "type": "function",
                                                        "function": {"name": fn.name,
                                                                     "arguments": fn.arguments}})


class FakeOverflow(Exception):
    """Stands in for openai.BadRequestError on a context-length error."""


TEXT = "__text__"        # step ("__text__", {"content": ..., "finish": ...}): reply without a call
OVERFLOW = "__overflow__"  # step that first fails with a context-length error, then calls a tool


class FakeClient:
    """Returns one scripted reply per request and records every prompt it was sent."""

    def __init__(self, script, seen=None):
        self._script = iter(script)
        self._overflowed = False
        self.seen = seen if seen is not None else []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        i, (name, args) = next(self._script)
        if name == OVERFLOW and not self._overflowed:
            self._overflowed = True
            self._script = iter([(i, (name, args))] + list(self._script))
            raise FakeOverflow("maximum context length exceeded")
        self.seen.append(kwargs["messages"])
        if name == TEXT:
            msg = SimpleNamespace(content=args["content"], tool_calls=None)
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=msg, finish_reason=args["finish"])])
        name = "bash" if name == OVERFLOW else name
        msg = SimpleNamespace(content="", tool_calls=[_call(i, name, **args)])
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")])


def _episode(tmp_path, monkeypatch, steps, max_turns=30, time_budget_sec=1800, seen=None):
    """Run a scripted episode; return (stats, tool events, start event)."""
    script = list(enumerate(steps + [("submit", {})]))
    monkeypatch.setattr(agent_mod, "OpenAI", lambda **_kw: FakeClient(script, seen))
    monkeypatch.setattr(agent_mod, "BadRequestError", FakeOverflow)
    work = tmp_path / "ws"
    work.mkdir()
    (work / "check.py").write_text("print('ok')\n")
    config = AgentConfig("http://unused", "fake", max_turns=max_turns, repeat_guard=True,
                         time_budget_sec=time_budget_sec)
    traj = tmp_path / "traj.jsonl"
    stats = run_agent("task", Sandbox(work), config, traj)
    events = [json.loads(l) for l in traj.open()]
    return stats, [e for e in events if e["type"] == "tool"], events[0]


def _run(tmp_path, monkeypatch, steps):
    """Which tool results carried a repeat note."""
    _stats, tools, _start = _episode(tmp_path, monkeypatch, steps)
    return ["exact tool call" in t["output"] for t in tools]


CHECK = ("bash", {"command": "python check.py"})


def _write(text):
    return ("write_file", {"path": "align.py", "content": text})


def test_rerunning_check_after_each_edit_is_not_flagged(tmp_path, monkeypatch):
    steps = []
    for v in range(4):
        steps += [_write(f"v = {v}\n"), CHECK]
    assert not any(_run(tmp_path, monkeypatch, steps))


def test_same_call_on_unchanged_workspace_is_flagged_on_third(tmp_path, monkeypatch):
    notes = _run(tmp_path, monkeypatch, [_write("v = 1\n"), CHECK, CHECK, CHECK])
    assert notes[:4] == [False, False, False, True]


def test_rewriting_identical_content_is_flagged(tmp_path, monkeypatch):
    # The first write changes the workspace; writes 2-4 start from the same state
    notes = _run(tmp_path, monkeypatch, [_write("v = 1\n")] * 4)
    assert notes[:4] == [False, False, False, True]


def test_ignoring_repeat_notes_ends_the_episode(tmp_path, monkeypatch):
    # Notes start on the 3rd identical call; the 5th ignored note in a row stops the episode
    stats, tools, _ = _episode(tmp_path, monkeypatch, [CHECK] * 12)
    assert stats["stop_reason"] == "repeat_loop"
    assert len(tools) == 2 + agent_mod.LOOP_STOP_NOTES
    assert not stats["submitted"]


def test_repeating_the_same_plan_without_tools_ends_the_episode(tmp_path, monkeypatch):
    plan = (TEXT, {"content": "I will now implement the estimator.", "finish": "stop"})
    stats, _tools, _ = _episode(tmp_path, monkeypatch, [plan] * 12)
    assert stats["stop_reason"] == "repeat_loop"
    events = [json.loads(l) for l in (tmp_path / "traj.jsonl").open()]
    feedback = [e["content"] for e in events if e["type"] == "user"]
    assert "same message 3 times" in feedback[2] and "same message" not in feedback[1]
    assert len(feedback) == 2 + agent_mod.LOOP_STOP_NOTES


def test_an_edit_resets_the_loop_counter(tmp_path, monkeypatch):
    steps = [CHECK] * 6 + [_write("v = 2\n")] + [CHECK] * 4
    stats, _tools, _ = _episode(tmp_path, monkeypatch, steps)
    assert stats["stop_reason"] == "submitted"


def test_budget_is_visible_to_the_agent(tmp_path, monkeypatch):
    stats, tools, start = _episode(tmp_path, monkeypatch, [CHECK], max_turns=30)
    assert "at most 30 turns" in start["messages"][0]["content"]
    assert tools[0]["output"].startswith("[turn 1/30, ")
    assert "Only" not in tools[0]["output"]  # plenty of budget left: no wrap-up note
    assert stats["stop_reason"] == "submitted"


def test_wrap_up_note_near_the_turn_limit(tmp_path, monkeypatch):
    _stats, tools, _ = _episode(tmp_path, monkeypatch, [_write(f"v = {i}\n") for i in range(8)],
                                max_turns=10)
    notes = ["Only" in t["output"] for t in tools]
    assert notes[:8] == [False] * 4 + [True] * 4  # turns 5..8 leave <= 5 turns
    assert "Only" not in tools[-1]["output"]       # not appended to submit


def test_command_timeout_never_exceeds_time_left(tmp_path):
    sb = Sandbox(tmp_path)
    out = agent_mod._call_tool(sb, "bash", {"command": "sleep 30", "timeout": 600}, max_timeout=1)
    assert out.startswith("[timeout after 1s:") and "larger `timeout`" in out


def test_feedback_explains_why_nothing_ran():
    fb = agent_mod.no_tool_feedback
    assert "cut off" in fb("x" * 100, "length", 4096)
    bad = fb('<tool_call>\n{"name": "bash", "arguments": {"command": "grep \\d x"}}\n</tool_call>', "stop", 4096)
    assert "could not be parsed" in bad and "Invalid \\escape" in bad
    assert "not saved" in fb("Here is the code:\n```python\nx = 1\n```", "stop", 4096)
    assert fb("I am done thinking.", "stop", 4096) == agent_mod.CONTINUE_PROMPT


def test_overflow_view_elides_old_prose_but_keeps_recent():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    for k in range(4):
        msgs += [{"role": "assistant", "content": f"plan {k} " + "p" * 500,
                  "tool_calls": [{"id": f"c{k}", "type": "function",
                                  "function": {"name": "bash", "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": f"c{k}", "content": "o" * 500}]
    view = agent_mod.compact_view(msgs, 2, elide_prose=True)
    assert view[1]["content"] == "task"                       # instruction never elided
    assert view[2]["content"].startswith("[earlier message")  # old reasoning elided
    assert view[-2]["content"].startswith("plan 3")           # recent reasoning kept
    assert agent_mod.compact_view(msgs, 2)[2]["content"].startswith("plan 0")


def test_sft_rebuild_matches_exactly_what_the_model_saw(tmp_path, monkeypatch):
    from training.build_sft_data import is_training_target, rebuild_turns
    seen = []
    steps = [_write("v = 1\n"),
             (TEXT, {"content": "x" * 50, "finish": "length"}),
             (TEXT, {"content": '<tool_call>\n{"name": "bash", "arguments": {"command": "\\d"}}\n</tool_call>',
                     "finish": "stop"}),
             (TEXT, {"content": "```python\nx = 1\n```", "finish": "stop"}),
             CHECK,
             (OVERFLOW, {"command": "python check.py"}),
             CHECK]
    _episode(tmp_path, monkeypatch, steps, seen=seen)
    turns = list(rebuild_turns(tmp_path / "traj.jsonl"))
    assert len(turns) == len(seen)
    for (history, event), prompt in zip(turns, seen):
        assert agent_mod.compact_view(history, *event["view"]) == prompt
    targets = [is_training_target(e) for _h, e in turns]
    assert targets == [True, False, False, False, True, True, True, True]


def test_fingerprint_tracks_content_not_just_names(tmp_path):
    sb = Sandbox(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    before = sb.fingerprint()
    (tmp_path / "a.py").write_text("x = 2\n")
    assert sb.fingerprint() != before
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.pyc").write_bytes(b"\0")
    after_edit = sb.fingerprint()
    (tmp_path / "__pycache__" / "a.pyc").write_bytes(b"\1")
    assert sb.fingerprint() == after_edit  # bytecode caches are ignored


def test_run_in_group_reports_timeout(tmp_path):
    code, _out, _err = run_in_group([sys.executable, "-c", "import time; time.sleep(30)"],
                                    timeout=1, text=True)
    assert code is None
