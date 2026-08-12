from __future__ import annotations

import random

from .models import Genome


def mutate_genome(genome: Genome, randomizer: random.Random) -> Genome:
    values = genome.model_dump()
    mutation_rate = genome.mutation_rate
    for name, current in values.items():
        if name == "mutation_rate":
            continue
        if randomizer.random() < mutation_rate:
            values[name] = min(1.0, max(0.0, current + randomizer.uniform(-0.05, 0.05)))
    return Genome.model_validate(values)

