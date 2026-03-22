from __future__ import annotations

import ast
from pathlib import Path


def parse_ok(path: Path) -> tuple[bool, str]:
    try:
        src = path.read_text(encoding="utf-8")
        ast.parse(src)
        return True, ""
    except SyntaxError as e:
        return False, f"{e}"
    except OSError as e:
        return False, str(e)


def parse_src_ok(source: str) -> tuple[bool, str]:
    """Validate Python source string without writing it to disk."""
    try:
        ast.parse(source)
        return True, ""
    except SyntaxError as e:
        return False, f"{e}"
