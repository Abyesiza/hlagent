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


def git_revert_to(repo: Path, commit_hash: str) -> tuple[bool, str]:
    """Hard-reset the working tree and index to a specific commit."""
    try:
        subprocess.run(
            ["git", "reset", "--hard", commit_hash],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return True, f"Reverted to {commit_hash[:8]}"
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        err = getattr(e, "stderr", str(e)) or str(e)
        return False, err


def git_log(repo: Path, limit: int = 10) -> list[dict[str, str]]:
    """Return the last `limit` commits as a list of {hash, subject, date}."""
    try:
        r = subprocess.run(
            ["git", "log", f"-{limit}", "--pretty=format:%H|%s|%ai"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        entries = []
        for line in r.stdout.strip().splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3:
                entries.append({"hash": parts[0], "subject": parts[1], "date": parts[2]})
        return entries
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
