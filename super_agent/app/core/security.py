from __future__ import annotations

import re
from typing import Final

_DANGEROUS_SHELL: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r";\s*rm\s+"),
    re.compile(r"\brm\s+-rf\b"),
    re.compile(r"mkfs\."),
    re.compile(r"dd\s+if="),
    re.compile(r">\s*/dev/"),
    re.compile(r"curl\s+.*\|\s*(ba)?sh"),
    re.compile(r"wget\s+.*\|\s*(ba)?sh"),
    re.compile(r"chmod\s+[-+]x\s+/"),
)


def is_shell_command_blocked(command: str) -> bool:
    normalized = command.strip().lower()
    if not normalized:
        return True
    return any(p.search(normalized) for p in _DANGEROUS_SHELL)


def validate_api_key_configured(api_key: str | None) -> bool:
    return bool(api_key and api_key.strip())
