"""Agent-side execution sandbox for docker-less Linux hosts.

When `user` is given (runner started as root), every agent command runs as that
unprivileged user inside the task workspace, and the task's hidden directories (tests/,
solution/, baselines/) are made root-only. Without a user (e.g. local Windows development)
commands run as the current user and isolation is NOT enforced; the runner records this
in the result as `"isolated": false`.
"""

import contextlib
import hashlib
import os
import shutil
import signal
import subprocess
import threading
from pathlib import Path

HIDDEN_DIRS = ("tests", "solution", "baselines")
MAX_OUTPUT_CHARS = 6000
# Per-process limits for agent commands so that one runaway program cannot take the host down
# (virtual memory in KiB for `ulimit -v`, and max processes of the sandbox user).
MEMORY_LIMIT_KB = 16 * 1024 * 1024
MAX_USER_PROCESSES = 4096
LIMITS_PREFIX = f"ulimit -v {MEMORY_LIMIT_KB}; ulimit -u {MAX_USER_PROCESSES}; "
# Numeric libraries must not spawn one thread per core: many trials share the host
SINGLE_THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

# CPU-bound work (agent commands, grading) holds a slot; waiting for the model does not.
# This decouples the number of concurrent trials from the number of busy cores.
_cpu_slots = None


def set_cpu_slots(n):
    """Allow at most n concurrent CPU-bound steps across all trials (0 = unlimited)."""
    global _cpu_slots
    _cpu_slots = threading.BoundedSemaphore(n) if n > 0 else None


@contextlib.contextmanager
def cpu_slot():
    if _cpu_slots is None:
        yield
        return
    with _cpu_slots:
        yield


def _kill_group(pgid):
    if hasattr(os, "killpg"):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)


def run_in_group(argv, timeout=None, **kwargs):
    """subprocess.run in a new process group that is killed afterwards, also on timeout.

    subprocess.run only kills the direct child on timeout, so grandchildren (the agent's
    `python check.py` under runuser/bash) used to survive and keep burning CPU.
    Returns (returncode or None on timeout, stdout, stderr).
    """
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=(os.name != "nt"), **kwargs)
    try:
        out, err = proc.communicate(timeout=timeout)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_group(proc.pid)
        proc.kill()
        out, err = proc.communicate()
        code = None
    finally:
        _kill_group(proc.pid)  # background jobs left behind by the command
    return code, out, err


def _proc_uid_ppid(proc_dir):
    uid = ppid = None
    for line in (proc_dir / "status").read_text().splitlines():
        if line.startswith("Uid:"):
            uid = int(line.split()[1])
        elif line.startswith("PPid:"):
            ppid = int(line.split()[1])
    return uid, ppid


def kill_leftovers(path, user=None):
    """Kill processes that outlived their command or trial (Linux only); return how many.

    Matches processes whose command line mentions path, and orphans of the sandbox user
    (re-parented to PID 1 after escaping their process group with setsid/double fork):
    a live agent command always has the runner as an ancestor. /proc/<pid>/cwd is not
    used because containers often deny reading it for other users' processes.
    """
    root, path = Path("/proc"), str(Path(path).resolve())
    if not root.is_dir():
        return 0
    uid = None
    if user:
        import pwd  # POSIX only
        uid = pwd.getpwnam(user).pw_uid
    killed = 0
    for proc_dir in root.iterdir():
        if not proc_dir.name.isdigit() or int(proc_dir.name) == os.getpid():
            continue
        try:
            cmdline = (proc_dir / "cmdline").read_bytes().decode(errors="replace")
            proc_uid, ppid = _proc_uid_ppid(proc_dir)
        except OSError:
            continue
        if not cmdline:  # zombie or kernel thread
            continue
        if path in cmdline or (uid is not None and proc_uid == uid and ppid == 1):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(int(proc_dir.name), signal.SIGKILL)
                killed += 1
    return killed


def lock_hidden_dirs(task_dir):
    """Make the whole task directory (tests, solution, guidance, ...) root-only.

    The agent receives its instruction and a copy of environment/workspace from the runner,
    so it never needs to read the task directory itself.
    """
    for name in HIDDEN_DIRS:
        path = Path(task_dir) / name
        if path.exists():
            os.chmod(path, 0o700)
    os.chmod(task_dir, 0o700)


def give_to_user(path, user):
    """Recursively chown a directory to the sandbox user (root only)."""
    shutil.chown(path, user=user, group=user)
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            shutil.chown(os.path.join(root, name), user=user, group=user)


def truncate(text, limit=MAX_OUTPUT_CHARS):
    if len(text) <= limit:
        return text
    head = limit // 3
    return text[:head] + f"\n... [{len(text) - limit} chars truncated] ...\n" + text[-(limit - head):]


class Sandbox:
    def __init__(self, workdir, user=None, python_bin=None):
        self.workdir = Path(workdir).resolve()
        self.user = user
        # Directory holding the python interpreter the agent should get as `python`
        self.python_dir = str(Path(python_bin).parent) if python_bin else None

    @property
    def isolated(self):
        return self.user is not None

    def _env(self):
        path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
        if self.python_dir:
            path = self.python_dir + os.pathsep + path
        env = {"PATH": path, "HOME": str(self.workdir), "LANG": "C.UTF-8",
               "PYTHONUNBUFFERED": "1", **SINGLE_THREAD_ENV}
        if os.name == "nt":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
        return env

    def run(self, command, timeout_sec=120):
        """Run a bash command in the workspace; return a printable result string."""
        argv = ["bash", "-c", command]
        if self.user:
            argv = ["runuser", "-u", self.user, "--", "bash", "-c", LIMITS_PREFIX + command]
        with cpu_slot():
            code, stdout, stderr = run_in_group(argv, timeout=timeout_sec, cwd=self.workdir,
                                                env=self._env(), text=True, errors="replace")
        if code is None:
            return (f"[timeout after {timeout_sec}s: the command was killed. If it legitimately needs "
                    f"longer, pass a larger `timeout` (max 600).]\n" + truncate(stdout or ""))
        out = stdout + (("\n[stderr]\n" + stderr) if stderr else "")
        return f"[exit code {code}]\n" + truncate(out)

    def fingerprint(self, max_file_bytes=4 * 1024 * 1024):
        """Hash of the workspace contents (paths, sizes, file bytes up to max_file_bytes).

        Lets the repeat guard tell a re-run after an edit (fine) from a true repeat.
        """
        h = hashlib.sha1()
        for root, dirs, files in os.walk(self.workdir):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for name in sorted(files):
                path = Path(root) / name
                try:
                    size = path.stat().st_size
                    h.update(f"{path.relative_to(self.workdir)}\0{size}\0".encode())
                    if size <= max_file_bytes:
                        h.update(path.read_bytes())
                except OSError:  # vanished or unreadable mid-walk
                    continue
        return h.hexdigest()

    def resolve(self, rel_path):
        """Map an agent-supplied path into the workspace, refusing escapes."""
        path = (self.workdir / rel_path).resolve()
        if path != self.workdir and self.workdir not in path.parents:
            raise PermissionError(f"path outside the workspace: {rel_path}")
        return path

    def write_file(self, rel_path, content):
        path = self.resolve(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        if self.user:  # chown the file and any parent directories created above
            for p in [path, *path.parents]:
                if p == self.workdir:
                    break
                shutil.chown(p, user=self.user, group=self.user)
        return f"wrote {len(content)} chars to {path.relative_to(self.workdir)}"

    def read_file(self, rel_path):
        path = self.resolve(rel_path)
        if not path.is_file():
            return f"no such file: {rel_path}"
        return truncate(path.read_text(errors="replace"))
