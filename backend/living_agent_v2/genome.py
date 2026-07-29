from __future__ import annotations

import random

from .models import GenomeV2


GENE_FIELDS = (
    "curiosity",
    "compression_bias",
    "symbolic_precision",
    "sociality",
    "mutation_rate",
    "energy_efficiency",
)


def random_genome(rng: random.Random) -> GenomeV2:
    return GenomeV2(
        curiosity=rng.uniform(0.25, 0.85),
        compression_bias=rng.uniform(0.2, 0.9),
        symbolic_precision=rng.uniform(0.25, 0.9),
        sociality=rng.uniform(0.1, 0.8),
        mutation_rate=rng.uniform(0.02, 0.18),
        energy_efficiency=rng.uniform(0.25, 0.85),
    )


def mutate_genome(parent: GenomeV2, rng: random.Random) -> GenomeV2:
    values = parent.model_dump()
    mutation_rate = parent.mutation_rate

    for field in GENE_FIELDS:
        if rng.random() <= mutation_rate:
            values[field] = _clamp(values[field] + rng.uniform(-0.08, 0.08))

    values["mutation_rate"] = _clamp(values["mutation_rate"], 0.01, 0.5)
    return GenomeV2(**values)


def inherit_genome(parent_a: GenomeV2, parent_b: GenomeV2, rng: random.Random) -> GenomeV2:
    values = {}
    for field in GENE_FIELDS:
        values[field] = (getattr(parent_a, field) + getattr(parent_b, field)) / 2
    return mutate_genome(GenomeV2(**values), rng)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))

