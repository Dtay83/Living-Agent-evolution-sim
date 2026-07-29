from __future__ import annotations

import math

from .models import AgentV2, CellV2, InfodynamicMetrics


def calculate_metrics(grid: list[list[CellV2]], agents: list[AgentV2]) -> InfodynamicMetrics:
    densities = [cell.information_density for row in grid for cell in row]
    resources = [cell.resource for row in grid for cell in row]
    energies = [agent.energy for agent in agents]

    entropy = _normalized_entropy(densities + resources + energies)
    compression_target = _compression_target(agents)
    compression_delta = max(0.0, entropy - compression_target)
    novelty = _novelty_score(grid, agents)
    information_pressure = compression_delta + novelty

    return InfodynamicMetrics(
        entropy=round(entropy, 6),
        compression_delta=round(compression_delta, 6),
        novelty=round(novelty, 6),
        information_pressure=round(information_pressure, 6),
    )


def evolve_cell(cell: CellV2, local_pressure: float) -> CellV2:
    density = max(0.0, cell.information_density * (1.0 - min(local_pressure, 0.2)))
    resource = max(0.0, cell.resource * 0.98)
    return cell.model_copy(update={"information_density": density, "resource": resource})


def movement_cost(agent: AgentV2, pressure: float) -> float:
    compression_discount = 0.35 * agent.genome.compression_bias
    efficiency_discount = 0.35 * agent.genome.energy_efficiency
    base_cost = 1.0 + pressure
    return max(0.15, base_cost * (1.0 - compression_discount - efficiency_discount))


def _normalized_entropy(values: list[float]) -> float:
    total = sum(max(0.0, value) for value in values)
    if total <= 0:
        return 0.0

    entropy = 0.0
    for value in values:
        if value <= 0:
            continue
        probability = value / total
        entropy -= probability * math.log2(probability)

    max_entropy = math.log2(len(values)) if values else 1.0
    if max_entropy == 0:
        return 0.0
    return entropy / max_entropy


def _compression_target(agents: list[AgentV2]) -> float:
    if not agents:
        return 0.0
    avg_bias = sum(agent.genome.compression_bias for agent in agents) / len(agents)
    return 0.8 - (avg_bias * 0.35)


def _novelty_score(grid: list[list[CellV2]], agents: list[AgentV2]) -> float:
    occupied = {(agent.x, agent.y) for agent in agents}
    density_cells = sum(1 for row in grid for cell in row if cell.information_density > 0.35)
    denominator = max(1, len(grid) * len(grid[0]))
    return min(1.0, (len(occupied) / denominator) + (density_cells / denominator))

