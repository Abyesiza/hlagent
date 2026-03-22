from __future__ import annotations

from pathlib import Path


def _tail_text(path: Path, max_chars: int = 4000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def load_soul(data_dir: Path, max_chars: int = 3000) -> str:
    return _tail_text(data_dir / "SOUL.md", max_chars)


def load_memory_excerpt(data_dir: Path, max_chars: int = 4000) -> str:
    return _tail_text(data_dir / "MEMORY.md", max_chars)


def load_codebase_snapshot(data_dir: Path, max_chars: int = 8000) -> str:
    """Load CODEBASE.md; auto-generate if missing."""
    path = data_dir / "CODEBASE.md"
    if not path.is_file():
        try:
            from super_agent.app.services.codebase_scanner import refresh_codebase_md
            refresh_codebase_md(data_dir)
        except Exception:
            return ""
    return _tail_text(path, max_chars)


def build_system_preamble(data_dir: Path) -> str:
    soul = load_soul(data_dir)
    mem = load_memory_excerpt(data_dir)
    codebase = load_codebase_snapshot(data_dir)

    parts: list[str] = []
    if soul:
        parts.append("## Agent Identity & Directives\n" + soul)
    if codebase:
        parts.append("## Codebase\n" + codebase)
    if mem:
        parts.append("## Memory (recent)\n" + mem)
    return "\n\n".join(parts).strip()
