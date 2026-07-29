from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass


VECTOR_SIZE = 384


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    agent_id: str
    concept_type: str
    vector: tuple[float, ...]
    raw_text: str | None


class MemoryStore:
    def __init__(self, keep_raw_text: bool = True):
        self.keep_raw_text = keep_raw_text
        self._records: dict[str, MemoryRecord] = {}

    def add(self, agent_id: str, concept_type: str, text: str) -> MemoryRecord:
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            concept_type=concept_type,
            vector=tuple(embed_text(text)),
            raw_text=text if self.keep_raw_text else None,
        )
        self._records[record.id] = record
        return record

    def search(self, text: str, limit: int = 5) -> list[MemoryRecord]:
        query = embed_text(text)
        scored = sorted(
            self._records.values(),
            key=lambda record: cosine_similarity(query, record.vector),
            reverse=True,
        )
        return scored[:limit]

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self._records.get(memory_id)


def embed_text(text: str) -> list[float]:
    vector = [0.0] * VECTOR_SIZE
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % VECTOR_SIZE
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    numerator = sum(left * right for left, right in zip(a, b))
    a_norm = math.sqrt(sum(value * value for value in a))
    b_norm = math.sqrt(sum(value * value for value in b))
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return numerator / (a_norm * b_norm)

