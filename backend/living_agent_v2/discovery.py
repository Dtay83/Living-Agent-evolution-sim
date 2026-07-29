from __future__ import annotations

import re
import uuid

try:
    import sympy as sp
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments
    sp = None

from .models import AgentV2, ArbitrationResult, Discovery, DiscoveryStatus


ENERGY_PATTERN = re.compile(r"energy\s*(?:=|equals)\s*(?P<value>-?\d+(?:\.\d+)?)", re.I)
MASS_PATTERN = re.compile(r"mass\s*(?:=|equals)\s*(?P<value>-?\d+(?:\.\d+)?)", re.I)
C_PATTERN = re.compile(r"\bc\s*(?:=|equals)\s*(?P<value>-?\d+(?:\.\d+)?)", re.I)


def propose_hypothesis(agent: AgentV2) -> str:
    mass = max(1, round(2 + agent.genome.symbolic_precision * 8))
    c_value = 10
    energy = mass * c_value**2
    if agent.genome.curiosity < 0.35:
        energy += 1
    return f"energy equals {energy}, mass equals {mass}, c equals {c_value}"


def arbitrate_hypothesis(hypothesis: str) -> ArbitrationResult:
    energy = _extract(ENERGY_PATTERN, hypothesis)
    mass = _extract(MASS_PATTERN, hypothesis)
    c_value = _extract(C_PATTERN, hypothesis) or 10.0

    if energy is None or mass is None:
        return ArbitrationResult(valid=False, feedback="Hypothesis must include numeric energy and mass.")

    if sp is None:
        residual_value = energy - (mass * c_value**2)
    else:
        E, m, c = sp.symbols("E m c")
        residual = sp.simplify(E - (m * c**2)).subs({E: energy, m: mass, c: c_value})
        residual_value = float(residual)

    if abs(residual_value) <= 1e-9:
        return ArbitrationResult(
            valid=True,
            feedback="Symbolic arbitration passed: E = m*c^2 resolves.",
            residual=residual_value,
        )

    return ArbitrationResult(
        valid=False,
        feedback="Symbolic arbitration failed: E = m*c^2 does not balance.",
        residual=residual_value,
    )


def make_discovery(agent: AgentV2, tick: int, hypothesis: str) -> Discovery:
    result = arbitrate_hypothesis(hypothesis)
    return Discovery(
        id=str(uuid.uuid4()),
        agent_id=agent.id,
        tick=tick,
        hypothesis=hypothesis,
        status=DiscoveryStatus.ACCEPTED if result.valid else DiscoveryStatus.REJECTED,
        feedback=result.feedback,
        residual=result.residual,
    )


def _extract(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    if not match:
        return None
    return float(match.group("value"))
