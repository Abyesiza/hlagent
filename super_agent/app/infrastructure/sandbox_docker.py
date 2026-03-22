"""
Sandbox execution: subprocess-based (always available) with optional Docker.

Priority:
  1. Docker  — if settings.enable_docker_sandbox=True and `docker` CLI is reachable
  2. subprocess  — isolated Python via `python -I`, always available
"""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from super_agent.app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    backend: str          # "subprocess" or "docker"
    timed_out: bool = False


def run_subprocess_sandbox(code: str, timeout: int = 15) -> SandboxResult:
    """
    Execute Python code in an isolated subprocess.

    Uses `python -I` (isolated mode: ignores PYTHONPATH, user site-packages,
    and the current directory). Works on any machine without Docker.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(code)
        tmp = Path(fh.name)
    try:
        result = subprocess.run(
            [sys.executable, "-I", str(tmp)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return SandboxResult(
            returncode=result.returncode,
            stdout=result.stdout[:8000],
            stderr=result.stderr[:4000],
            backend="subprocess",
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            returncode=-1, stdout="", stderr=f"Execution timed out after {timeout}s",
            backend="subprocess", timed_out=True,
        )
    except Exception as e:
        logger.exception("subprocess sandbox error")
        return SandboxResult(returncode=-1, stdout="", stderr=str(e), backend="subprocess")
    finally:
        tmp.unlink(missing_ok=True)


def run_docker_sandbox(
    settings: Settings,
    command: list[str],
    workspace_host_path: Path,
    timeout: int = 60,
) -> SandboxResult:
    """
    Run a command in Docker with the workspace mounted.
    Falls back gracefully if docker CLI is missing or fails.
    """
    if not settings.enable_docker_sandbox:
        return SandboxResult(
            returncode=-1, stdout="", stderr="Docker sandbox disabled",
            backend="docker",
        )
    image = settings.docker_sandbox_image
    ws = workspace_host_path.resolve()
    try:
        result = subprocess.run(
            ["docker", "run", "--rm",
             "--network=none",           # no outbound network
             "--memory=256m",
             "--cpus=0.5",
             "-v", f"{ws}:/workspace:ro",
             "-w", "/workspace",
             image, *command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return SandboxResult(
            returncode=result.returncode,
            stdout=result.stdout[:8000],
            stderr=result.stderr[:4000],
            backend="docker",
        )
    except FileNotFoundError:
        return SandboxResult(returncode=-1, stdout="", stderr="docker CLI not found", backend="docker")
    except subprocess.TimeoutExpired:
        return SandboxResult(
            returncode=-1, stdout="", stderr=f"Docker timed out after {timeout}s",
            backend="docker", timed_out=True,
        )
    except Exception as e:
        logger.exception("docker sandbox error")
        return SandboxResult(returncode=-1, stdout="", stderr=str(e), backend="docker")


def run_sandbox(settings: Settings, code: str, timeout: int = 15) -> SandboxResult:
    """
    Route to Docker if enabled and available, otherwise subprocess.
    This is the main entry point used by the API.
    """
    if settings.enable_docker_sandbox:
        result = run_docker_sandbox(
            settings,
            ["python", "-c", code],
            settings.sandbox_dir,
            timeout=timeout,
        )
        if result.returncode != -1 or "not found" not in result.stderr:
            return result
        logger.warning("Docker unavailable, falling back to subprocess sandbox")
    return run_subprocess_sandbox(code, timeout)
