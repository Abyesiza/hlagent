"""
Subprocess-only code sandbox.

Docker has been removed in favour of a simpler, always-available approach:
  - python -I  (isolated mode: ignores PYTHONPATH, user site-packages)
  - resource limits via a tight timeout
  - stdout / stderr captured and truncated to prevent output flooding
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from super_agent.app.core.config import Settings


@dataclass
class SandboxResult:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    backend: str = "subprocess"
    timed_out: bool = False


_MAX_OUTPUT = 8_000   # chars — truncate large output so the API response stays reasonable


def run_sandbox(settings: "Settings", code: str, timeout: int = 15) -> SandboxResult:
    """
    Execute *code* in an isolated subprocess and return the result.

    The code is written to a temp file and run with ``python -I`` which:
      - ignores PYTHONPATH
      - ignores the user site directory
      - prevents importing site-packages that could escape the sandbox

    Network access is NOT blocked at the OS level here; that responsibility
    belongs to the caller or a future seccomp/nsjail wrapper.
    """
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", encoding="utf-8", delete=False) as f:
        f.write(textwrap.dedent(code))
        tmp = Path(f.name)

    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(tmp)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = proc.stdout[:_MAX_OUTPUT]
        stderr = proc.stderr[:_MAX_OUTPUT]
        return SandboxResult(
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            backend="subprocess",
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            returncode=1,
            stdout="",
            stderr=f"Execution timed out after {timeout}s.",
            backend="subprocess",
            timed_out=True,
        )
    except Exception as e:
        return SandboxResult(returncode=1, stderr=str(e), backend="subprocess")
    finally:
        tmp.unlink(missing_ok=True)
