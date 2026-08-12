from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def embed_text(text: str, dimensions: int = 64) -> list[float]:
    """Create a deterministic local feature-hash embedding.

    This is intentionally a lexical bootstrap vectorizer, not a semantic model.
    """
    if dimensions < 8:
        raise ValueError("dimensions must be at least 8")

    vector = [0.0] * dimensions
    for token in TOKEN_PATTERN.findall(text.casefold()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        index = int.from_bytes(digest[:8], "big") % dimensions
        sign = 1.0 if digest[8] & 1 else -1.0
        vector[index] += sign

    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have equal dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


@dataclass(frozen=True, slots=True)
class MemoryMatch:
    memory_id: str
    content: str
    score: float

