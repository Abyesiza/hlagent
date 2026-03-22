from __future__ import annotations

import re

from super_agent.app.domain.math_schemas import SymbolicIntent

_SYMBOL_HINTS = re.compile(
    r"\b(integrat|differentiat|derivative|lim\b|limit\b|solve\b|"
    r"equation|matrix|eigen|sum\b|∑|∫|∂|sympy|algebra|calculus|prove\b)\b",
    re.IGNORECASE,
)


def classify_intent(text: str) -> SymbolicIntent:
    if _SYMBOL_HINTS.search(text):
        return SymbolicIntent.SYMBOLIC
    # Heavier math notation (avoid routing "2+2" trivia to SymPy)
    if re.search(r"\*\*|//|∫|∂|∑|\^", text):
        return SymbolicIntent.SYMBOLIC
    return SymbolicIntent.NEURAL
