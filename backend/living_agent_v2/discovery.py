from __future__ import annotations

import re

from .models import ArbitrationResult


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
LABELED_VALUE = re.compile(
    rf"\b(?P<label>energy|mass|c)\s*(?:equals|=|is)\s*(?P<value>{NUMBER})\b",
    re.IGNORECASE,
)


def arbitrate_hypothesis(hypothesis: str) -> ArbitrationResult:
    values: dict[str, float] = {}
    for match in LABELED_VALUE.finditer(hypothesis):
        values[match.group("label").casefold()] = float(match.group("value"))

    missing = sorted({"energy", "mass", "c"} - values.keys())
    if missing:
        return ArbitrationResult(
            valid=False,
            feedback=f"Cannot arbitrate: missing {', '.join(missing)}.",
            residual=None,
        )

    if values["mass"] < 0 or values["c"] <= 0:
        return ArbitrationResult(
            valid=False,
            feedback="Cannot arbitrate: mass must be non-negative and c must be positive.",
            residual=None,
        )

    try:
        from sympy import Float, simplify

        residual = float(
            simplify(
                Float(str(values["energy"]))
                - Float(str(values["mass"])) * Float(str(values["c"])) ** 2
            )
        )
    except ImportError:
        residual = values["energy"] - values["mass"] * values["c"] ** 2

    tolerance = max(1e-9, abs(values["energy"]) * 1e-9)
    valid = abs(residual) <= tolerance
    feedback = (
        "Symbolic arbitration passed: E = m*c^2 resolves."
        if valid
        else "Symbolic arbitration rejected: E does not equal m*c^2."
    )
    return ArbitrationResult(valid=valid, feedback=feedback, residual=residual)

