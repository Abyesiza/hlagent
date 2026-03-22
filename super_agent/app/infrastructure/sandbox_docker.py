from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from super_agent.app.core.config import Settings

logger = logging.getLogger(__name__)


def run_in_docker(
    settings: Settings,
    command: list[str],
    workspace_host_path: Path,
) -> tuple[int, str, str]:
    """
    Optional Docker isolation. Requires Docker daemon and `enable_docker_sandbox`.
    """
    if not settings.enable_docker_sandbox:
        return -1, "", "docker sandbox disabled (set SUPER_AGENT_ENABLE_DOCKER_SANDBOX=true)"
    image = settings.docker_sandbox_image
    ws = workspace_host_path.resolve()
    try:
        r = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{ws}:/workspace",
                "-w",
                "/workspace",
                image,
                *command,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return -1, "", "docker CLI not found"
    except subprocess.TimeoutExpired:
        return -1, "", "docker run timeout"
    except Exception as e:
        logger.exception("docker sandbox")
        return -1, "", str(e)
