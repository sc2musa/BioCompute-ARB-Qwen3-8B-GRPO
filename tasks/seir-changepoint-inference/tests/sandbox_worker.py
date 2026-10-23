"""Run untrusted submission code in a separate process, isolated from hidden grading data.

Canonical copy lives in harness/common/; each task's tests/ directory holds an identical
copy so that tests/ is self-contained (Harbor verifier only mounts /tests).
`python scripts/sync_common.py --check` verifies the copies are identical.

The parent (grader) keeps expected answers in memory and sends only call arguments to the
child over stdin. If GRADER_SANDBOX_PREFIX is set (e.g. "runuser -u agent --"), the child
runs as an unprivileged user that cannot read tests/.
"""

import json
import os
import shlex
import subprocess
import sys
import tempfile

RESULT_MARKER = "<<<GRADER_RESULT>>>"

_CHILD_SOURCE = r'''
import importlib.util, json, sys, time, traceback
MARKER = "<<<GRADER_RESULT>>>"

def to_jsonable(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, float) and x != x:
        return None
    return x

def main():
    module_path, func_name = sys.argv[1], sys.argv[2]
    calls = json.loads(sys.stdin.read())
    sys.path.insert(0, __import__("os").path.dirname(module_path))
    results = []
    try:
        spec = importlib.util.spec_from_file_location("submission", module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, func_name)
    except Exception:
        err = traceback.format_exc(limit=3)
        print(MARKER + json.dumps([{"ok": False, "error": "import: " + err}] * len(calls)))
        return
    for kwargs in calls:
        t0 = time.perf_counter()
        try:
            value = to_jsonable(fn(**kwargs))
            results.append({"ok": True, "value": value, "seconds": time.perf_counter() - t0})
        except Exception:
            results.append({"ok": False, "error": traceback.format_exc(limit=3),
                            "seconds": time.perf_counter() - t0})
    sys.stdout.flush()
    print(MARKER + json.dumps(results))

main()
'''


def run_submission(module_path, func_name, calls, timeout_sec, workdir=None):
    """Call func_name(**kwargs) for each kwargs in calls inside a child process.

    Returns a list of {"ok", "value"|"error", "seconds"} dicts, one per call. On timeout or
    crash every call is reported as failed.
    """
    fd, child_path = tempfile.mkstemp(suffix="_grader_child.py", dir=workdir)
    with os.fdopen(fd, "w") as f:
        f.write(_CHILD_SOURCE)
    os.chmod(child_path, 0o755)
    prefix = shlex.split(os.environ.get("GRADER_SANDBOX_PREFIX", ""))
    cmd = prefix + [sys.executable, child_path, os.path.abspath(module_path), func_name]
    try:
        proc = subprocess.run(
            cmd, input=json.dumps(calls), capture_output=True, text=True,
            timeout=timeout_sec, cwd=workdir,
        )
    except subprocess.TimeoutExpired:
        return [{"ok": False, "error": f"timeout after {timeout_sec}s"} for _ in calls]
    finally:
        os.remove(child_path)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            results = json.loads(line[len(RESULT_MARKER):])
            if len(results) == len(calls):
                return results
            break
    err = (proc.stderr or "")[-2000:] or f"no result (exit code {proc.returncode})"
    return [{"ok": False, "error": err} for _ in calls]
