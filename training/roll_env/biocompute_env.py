"""ROLL / GEM environment for BioCompute-ARB tasks (multi-turn tool use, verifier reward).

Installed into ROLL as roll/pipeline/agentic/env/biocompute/ and registered as "biocompute"
(training/roll_patches/register_biocompute_env.py). Runs under AgentNativeStepEnvManager: every
observation is the full message list and the tool schemas go to the chat template through
info["tools"], so the prompt is rendered exactly like vLLM renders it at evaluation time.

An episode reproduces one evaluation run of benchmark configuration F (goal-only instruction,
no thinking, repeat guard, public check.py) by reusing harness/agent.py: the same system prompt
and budget, `[turn i/N, Ns left]` prefixes, repeat guard and loop stop, feedback for replies
without a usable tool call, low-budget note and compaction of old tool outputs. Model replies
are parsed like vLLM's hermes tool parser does. The episode ends on `submit`, the turn or time
budget, a repeat loop or a forced stop by the env manager; the workspace is then graded by the
task's independent verifier and reward = score + solve_bonus * solved (given at the last step).
"""

import json
import os
import re
import secrets
import shutil
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from gem import Env

# Same pattern as vLLM's Hermes2ProToolParser (an unterminated last call is still parsed)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.S)
END_TOKEN = "<|im_end|>"


def parse_reply(text):
    """(content, calls) as vLLM's hermes parser returns them; any bad call -> no calls."""
    if "<tool_call>" not in text:
        return text, []
    calls = []
    try:
        for closed, unclosed in TOOL_CALL_RE.findall(text):
            obj = json.loads(closed or unclosed)
            # Arguments as a JSON string, like the OpenAI-format messages of the evaluation harness
            # (harness.agent.compact_view parses them to elide old calls)
            arguments = json.dumps(obj.get("arguments", {}), ensure_ascii=False)
            calls.append({"id": f"call_{secrets.token_hex(8)}", "type": "function",
                          "function": {"name": obj["name"], "arguments": arguments}})
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return text, []
    return text[: text.find("<tool_call>")], calls


def as_tool_call(call):
    """Object view of a parsed call for harness.agent._ToolRunner."""
    fn = call["function"]
    return SimpleNamespace(id=call["id"], function=SimpleNamespace(**fn))


class BioComputeEnv(Env):
    def __init__(self, repo_dir: str, tasks_root: str, work_root: str,
                 task_list: Optional[str] = None, sandbox_user: str = "agent",
                 max_turns: int = 30, time_budget_sec: int = 1800, max_tokens: int = 4096,
                 solve_bonus: float = 0.5, episode_log: Optional[str] = None, **_: Any) -> None:
        super().__init__()
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        from harness import agent, runner, sandbox  # benchmark harness (same code as evaluation)
        self._agent, self._runner, self._sandbox_mod = agent, runner, sandbox
        self.tasks_root = Path(tasks_root)
        names = sorted(p.name for p in self.tasks_root.iterdir() if (p / "task.toml").exists())
        if task_list:
            wanted = set(Path(task_list).read_text().split())
            names = [n for n in names if n in wanted]
        if not names:
            raise ValueError("no tasks selected")
        self.task_names = names
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.sandbox_user = sandbox_user or None
        if self.sandbox_user:
            os.chmod(self.work_root, 0o711)  # the sandbox user must traverse into its workspace
            for name in names:
                sandbox.lock_hidden_dirs(self.tasks_root / name)
        self.config = agent.AgentConfig(base_url="", model="", max_turns=max_turns,
                                        max_tokens=max_tokens, time_budget_sec=time_budget_sec,
                                        repeat_guard=True)
        self.solve_bonus, self.episode_log = solve_bonus, episode_log
        self._log_lock = threading.Lock()
        self._trial_dir = None
        # Read by AgentNativeStepEnvManager
        self.env_reset_failed = False
        self.env_timeout = False

    # ------------------------------------------------------------------ episode
    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        self._cleanup()
        self.task = self.task_names[(0 if seed is None else int(seed)) % len(self.task_names)]
        task_dir = self.tasks_root / self.task
        self._trial_dir = self.work_root / f"{self.task}-{secrets.token_hex(6)}"
        self._trial_dir.mkdir(parents=True)
        app_dir = self._trial_dir / f"app-{secrets.token_hex(8)}"
        shutil.copytree(task_dir / "environment" / "workspace", app_dir)
        shutil.copy(task_dir / "public_check.py", app_dir / "check.py")
        if self.sandbox_user:
            os.chmod(self._trial_dir, 0o711)
            self._sandbox_mod.give_to_user(app_dir, self.sandbox_user)
        self._app_dir = app_dir
        sandbox = self._sandbox_mod.Sandbox(app_dir, user=self.sandbox_user, python_bin=sys.executable)
        agent, cfg = self._agent, self.config
        system = agent.SYSTEM_PROMPT + agent.BUDGET_PROMPT.format(
            turns=cfg.max_turns, minutes=round(cfg.time_budget_sec / 60))
        instruction = self._runner.build_instruction(task_dir, "goal", app_dir, public_check=True)
        self.messages = [{"role": "system", "content": system},
                         {"role": "user", "content": instruction}]
        self.stats = {"turns": 0, "tool_calls": 0, "tool_errors": 0, "submitted": False,
                      "stop_reason": "max_turns", "repeated_calls": 0, "repeat_notes": 0}
        self._start = time.time()
        self._tools = agent._ToolRunner(sandbox, cfg, self.stats, self._start)
        self.env_reset_failed = False
        return self._view(), {"tools": agent.TOOLS, "task_name": self.task}

    def step(self, action: Any):
        if not isinstance(action, str):  # EpisodeStopReason from the env manager (e.g. context full)
            self.stats["stop_reason"] = f"forced: {getattr(action, 'name', action)}"
            return self._finish()
        self.stats["turns"] += 1
        turn = self.stats["turns"]
        truncated = END_TOKEN not in action  # no end token: the reply hit max_new_tokens
        content, calls = parse_reply(action.split(END_TOKEN)[0])
        assistant = {"role": "assistant", "content": content}
        if calls:
            assistant["tool_calls"] = calls
        self.messages.append(assistant)
        if not calls:
            feedback = (self._agent.no_tool_feedback(content, "length" if truncated else "stop",
                                                     self.config.max_tokens)
                        + self._tools.text_reply(content))
            self.messages.append({"role": "user", "content": feedback})
        else:
            for call in calls:
                output = self._tools.run(as_tool_call(call), turn)
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
                if call["function"]["name"] == "submit":
                    self.stats["submitted"] = True
        if self.stats["submitted"]:
            self.stats["stop_reason"] = "submitted"
        elif self._tools.looping():
            self.stats["stop_reason"] = "repeat_loop"
        elif time.time() - self._start > self.config.time_budget_sec:
            self.stats["stop_reason"] = "time_budget"
        elif turn < self.config.max_turns:
            return self._view(), 0.0, False, False, self._info()
        return self._finish()

    # ------------------------------------------------------------------ helpers
    def _view(self):
        """What the model sees next: the harness' compacted copy of the conversation."""
        return self._agent.compact_view(self.messages, *self._agent.VIEW_DEFAULT)

    def _finish(self):
        task_dir = self.tasks_root / self.task
        self._sandbox_mod.kill_leftovers(self._app_dir, self.sandbox_user)
        solved, metrics = self._runner.verify(task_dir, self._trial_dir, self._app_dir, self.sandbox_user)
        score = float(metrics.get("score", 0.0))
        reward = score + self.solve_bonus * solved
        self.stats["wall_seconds"] = round(time.time() - self._start, 1)
        if self.episode_log:
            with self._log_lock, open(self.episode_log, "a") as f:
                f.write(json.dumps({"task": self.task, "score": score, "solved": int(solved),
                                    "reward": reward, **self.stats}) + "\n")
        info = self._info(score=score, solved=float(solved))
        observation = self._view()
        self._cleanup()
        return observation, reward, True, False, info

    def _info(self, **final):
        metrics = {"current_step": self.stats["turns"], "num_tool_calls": self.stats["tool_calls"],
                   "submitted": float(self.stats["submitted"]), **final}
        return {"metrics": metrics, "metrics_agg_mode": {k: "last" for k in metrics},
                "failure_mode": "", "error_messages": [], "test_output": "",
                "stop_reason": self.stats["stop_reason"]}

    @property
    def env_info(self):
        return {"task": getattr(self, "task", None), "turns": self.stats.get("turns", 0)
                if hasattr(self, "stats") else 0}

    def _cleanup(self):
        if self._trial_dir is not None:
            shutil.rmtree(self._trial_dir, ignore_errors=True)
            self._trial_dir = None

    def close(self):
        self._cleanup()
