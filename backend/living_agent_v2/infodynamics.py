from __future__ import annotations

import math
import zlib
from collections import Counter

from .models import InfodynamicMetrics


def shannon_entropy(values: list[int]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    length = len(values)
    return -sum(
        (count / length) * math.log2(count / length) for count in counts.values()
    )


def compression_ratio(payload: bytes) -> float:
    if not payload:
        return 0.0
    return len(zlib.compress(payload, level=9)) / len(payload)


def calculate_metrics(
    current_values: list[int], previous_values: list[int] | None = None
) -> InfodynamicMetrics:
    current_payload = bytes(current_values)
    entropy = shannon_entropy(current_values)
    current_ratio = compression_ratio(current_payload)
    previous_ratio = compression_ratio(bytes(previous_values or []))
    compression_delta = current_ratio - previous_ratio if previous_values else 0.0

    if previous_values:
        compared = min(len(current_values), len(previous_values))
        changed = sum(
            current_values[index] != previous_values[index]
            for index in range(compared)
        )
        changed += abs(len(current_values) - len(previous_values))
        novelty = changed / max(len(current_values), len(previous_values), 1)
    else:
        novelty = 0.0

    information_pressure = max(0.0, entropy * (1.0 + novelty) * current_ratio)
    return InfodynamicMetrics(
        entropy=entropy,
        compression_delta=compression_delta,
        novelty=novelty,
        information_pressure=information_pressure,
    )

