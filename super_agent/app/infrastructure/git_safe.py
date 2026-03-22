from __future__ import annotations

import subprocess
from pathlib import Path


def git_commit_all(repo: Path, message: str) -> tuple[bool, str]:
    try:
        subprocess.run(
            ["git", "add", "-A"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        r = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return False, r.stderr or r.stdout or "commit failed"
        return True, r.stdout
    except FileNotFoundError:
        return False, "git not installed"
    except subprocess.CalledProcessError as e:
        return False, e.stderr or str(e)


def current_head(repo: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return r.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
